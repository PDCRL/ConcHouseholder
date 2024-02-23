#include <iostream>
#include <cmath>
#include <climits>
#include <vector>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <atomic>


const int beta = 5;
std::atomic_int gctr;
std::atomic_int pivot_idx;
std::atomic<double> pivot_var;

void compute_transform(int num_threads, int m, int n, double **mat, double *pivots){

    int *pr = new int[m];
    for (int i = 0; i<m; i++) pr[i] = 0;

    //double pivot_var = mat[0][0];
    //int pivot_idx = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        double sm = 0, sm1 = 0, cl = 0, clinv = 0, up = 0, b = 0, d__1 = 0;
        int ctr = 0;
        double local_pivot_var = 0;
        int local_pivot_idx = 0;
        //int lpivot = 1;
        for(int lpivot = 1; lpivot <= m; lpivot++){

            local_pivot_idx = pivot_idx.load(std::memory_order_seq_cst);
            local_pivot_var = pivot_var.load(std::memory_order_seq_cst);
            if (local_pivot_idx != lpivot-1){
                local_pivot_var = mat[lpivot-1][lpivot-1];
            }

            #pragma omp for nowait private(sm) schedule(dynamic, beta)
            for (int j = 1; j <= m-lpivot; j++){

                if (ctr != lpivot){
                    ctr = lpivot;
                    
                    //#pragma omp critical
                    //std::cout<<"lpivot "<<lpivot<<" tid "<<omp_get_thread_num()<<" local_pivot_var "<<local_pivot_var<<'\n'; 

                    cl = std::fabs(local_pivot_var);
                    sm1 = 0.0;

                    for(std::size_t k = lpivot+1; k <= n; k++){
                        sm = std::fabs(mat[lpivot-1][k-1]);
                        sm1 += sm * sm;
                        cl = std::max(sm, cl);
                    }

                    if (cl <= 0.0) {continue;}

                    clinv = 1.0/cl;

                    d__1 = local_pivot_var * clinv;
                    sm = d__1 * d__1;
                    sm += sm1 * clinv * clinv;
                    cl  *= std::sqrt(sm);

                    if (local_pivot_var > 0.0) { cl = -cl; }

                    up = local_pivot_var - cl;
                    b = up * cl;
                    if (b >= 0.0) {continue;}
                    b = 1.0/b;


                    if (local_pivot_idx == lpivot-1)
                    {
                        int old = lpivot-1;
                        if (gctr.compare_exchange_strong(old, lpivot, std::memory_order_seq_cst)){
                            mat[lpivot-1][lpivot-1] = cl;
                        }
                    }
                            pivots[lpivot-1] = cl;
                }
                int cold = 0;
                while (1) {
                    #pragma omp atomic read seq_cst
                    cold = pr[lpivot-1+j];

                    if (cold == lpivot -1 ){
                        sm = mat[lpivot -1 +j][lpivot - 1] * up;

                        //std::cout<<"Before: "<<up<<','<<sm<<','<<std::endl;
                        for (std::size_t i__ = lpivot+1; i__ <= n; i__++) {
                            sm += mat[lpivot -1 +j][i__ - 1] * mat[lpivot - 1][i__ - 1];
                        }

                        if (sm == 0.0) {continue;}

                        sm *= b;
                        mat[lpivot -1 +j][lpivot -1] += sm * up;

                        for (std::size_t i__ = lpivot+1; i__ <= n; i__++) {
                            mat[lpivot -1 +j][i__ - 1] += sm * mat[lpivot - 1][i__ - 1];
                        }

                        #pragma omp atomic write seq_cst
                        pr[lpivot-1+j] = cold+1;
                        break;
                    }
                }
            }
            if (lpivot < m){
                int update_val = INT_MIN;
                while (1){
                    #pragma omp atomic read seq_cst
                    update_val = pr[lpivot];
                    if (update_val == lpivot){
                        #pragma omp single nowait
                        {
                            pivot_var.store(mat[lpivot][lpivot], std::memory_order_seq_cst);
                            //pivot_idx.store(lpivot, std::memory_order_seq_cst);
                            //pivots[lpivot] = mat[lpivot][lpivot];
                        }
                        //pivot_var = mat[lpivot][lpivot];
                        //lpivot++;
                        break;
                    }
                }
            }
            //else{
                //break;
            //}
        }
    }
    delete[] pr;

    for (int i = 0; i<m-1; i++)
        mat[i][i] = pivots[i];

}

void householder(int num_threads, const std::string& filename){
    std::cout<<"Num Threads: "<<num_threads<<std::endl;
    
    std::stringstream savefile;

    std::size_t m, n;
    std::ifstream infile(filename);
    infile >>m;
    infile >>n;
    
    double** temp_matrix = new double*[m];
    for (int i = 0; i<m; i++){
        temp_matrix[i] = new double[n];
    } 
    
    double *pivots = new double[m];

    for (std::size_t i = 0; i <m; i++){
        for(std::size_t j = 0; j < n; j++){
            infile >> temp_matrix[i][j];
            if (i == j){
                pivots[i] = temp_matrix[i][j];
            }
        }
    }

    gctr.store(0, std::memory_order_seq_cst);
    pivot_idx.store(0, std::memory_order_seq_cst);
    pivot_var.store(temp_matrix[0][0], std::memory_order_seq_cst);

    std::cout<<"Matrix Size: ("<<m<<", "<<n<<")\n";

    double tstart = omp_get_wtime();
    compute_transform(num_threads, m, n, temp_matrix, pivots);
    double tend = omp_get_wtime();

    std::cout<<"Finished Householder Transform\n";
    std::cout<<"Time Taken: "<<tend-tstart<<"s\n";
    
    savefile<<"QR_"<<m<<"x"<<n<<"_barrier_free_omp.txt";

    std::cout<<"Writing output to "<<savefile.str()<<std::endl;

    std::ofstream outfile(savefile.str());

    for (std::size_t i = 0; i < m; i++){
	for (std::size_t j = 0; j < n; j++){
	    if (j == n -1) outfile << temp_matrix[i][j];
	    else outfile<<temp_matrix[i][j]<<' ';
	}
    	outfile<<'\n';
    }

}

int main(int argc, char *argv[]){
    int num_threads = std::stoi(argv[1]);
    std::string filename = argv[2];
    householder(num_threads, filename);
    return 0;
}

