#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>
#include <pthread.h>
#include <fstream>
#include <sstream>
#include <atomic>
#include <mutex>
#include <climits>

struct args_t{
    int tid;
    std::size_t m;
    std::size_t n;
    double** matrix_ptr;
    double* up_arr;
    double* b_arr;
};

struct load{
    unsigned int iter;
    unsigned int start;
    unsigned int end;
    double pivot_var;
};

std::atomic<load> gload;

const unsigned int beta = 5;
std::atomic<int> global_ctr;

//std::atomic<std::atomic<int>*> pr;
//std::atomic<std::atomic<double*>*> br;

struct atomic_ptrs{
    int pr;
    double *br;
};

std::atomic<std::atomic<atomic_ptrs>*> ar;

//std::atomic<double*>* br;

std::mutex mtx;

void* thdwork(void* params){

    args_t* args = (struct args_t*)params;

    double sm = 0, sm1 = 0, cl = 0, clinv = 0, up = 0, b = 0, pivot_var = 0.0;
    
    auto temp_matrix = args->matrix_ptr;

    int ctr = 0;
    std::stringstream print_buffer;

    //double *new_ptr = nullptr, *old_ptr = nullptr;
    double* new_ptr = new double[args->n];
    int cold = 0;
    
    while(1){
        load old, local;
        old = gload.load(std::memory_order_seq_cst);
        local = old;

        local.start = old.end;
        local.end = old.end + beta;

        if(local.end > args->m - local.iter){
            local.end = args->m - local.iter +1;
	    }
        
        if(local.iter <= args->m && gload.compare_exchange_strong(old, local, std::memory_order_seq_cst)){
            int lpivot = local.iter;
            //double* pivot_row_ptr = br[lpivot-1].load(std::memory_order_seq_cst);
            atomic_ptrs local_ar = ar[lpivot-1].load(std::memory_order_seq_cst);
            double* pivot_row_ptr = local_ar.br;

            if (ctr != lpivot){
                ctr = lpivot;
		        pivot_var = local.pivot_var;

                cl = std::fabs(pivot_var);
                sm1 = 0.0;

                for(std::size_t k = lpivot+1; k <= args->n; k++){
                    //sm = std::fabs(temp_matrix[lpivot-1][k-1]);
                    sm = std::fabs(pivot_row_ptr[k-1]);
                    sm1 += sm * sm;
                    cl = std::max(sm, cl);
                }
        		//if (cl <= 0.0) {std::cout<<"cl <= 0 \n"; return;}

		        clinv = 1.0/cl;

		        double d__1 = pivot_var * clinv;
		        sm = d__1 * d__1;
		        sm += sm1 * clinv * clinv;
		        cl  *= std::sqrt(sm);

		        if (pivot_var > 0.0) { cl = -cl; }

		        up = pivot_var - cl;
                b = up * cl;
                //if (b >= 0.0) { std::cout<<"b >= 0 \n"; return;}

                b = 1.0/b;

                int old_ctr = lpivot-1;
                if(global_ctr.compare_exchange_strong(old_ctr, lpivot, std::memory_order_seq_cst)){
                    //temp_matrix[lpivot-1][lpivot-1] = cl;
                    pivot_row_ptr[lpivot-1] = cl;
                    args->up_arr[lpivot-1] = up;
                    args->b_arr[lpivot-1] = b;
                }
            }
            
            for(std::size_t j = local.start; j < local.end; j++){

			    //Write the helping part here
                while (1){

                    atomic_ptrs old_ar = ar[lpivot-1+j].load(std::memory_order_seq_cst);
                    double* old_ptr = old_ar.br;

                    //store the old value for this row in the local buffer
                    for (std::size_t i__ = 1; i__ <= args->n; i__++) {
                        new_ptr[i__ - 1] = old_ptr[i__ - 1];
                    }

                    //cold = pr[lpivot-1+j].load(std::memory_order_seq_cst);
                    cold = old_ar.pr;
                
                    if (cold < local.iter){

                        int cnew = cold+1;

                        double prev_up, prev_b, *prev_pivot_row_ptr;
                        
                        if (cnew == local.iter){
                            prev_up = up;
                            prev_b = b;
                            prev_pivot_row_ptr = pivot_row_ptr;
                        }
                        else{
                            prev_up = args->up_arr[cnew-1];
                            prev_b = args->b_arr[cnew-1];
                            atomic_ptrs much_old_ar = ar[cnew-1].load(std::memory_order_seq_cst);
                            prev_pivot_row_ptr = much_old_ar.br;
                        }

                        sm = new_ptr[cnew - 1] * prev_up;
        
                        for (int i__ = cnew+1; i__ <= args->n; i__++) {
                            //sm += temp_matrix[lpivot -1 +j][i__ - 1] * temp_matrix[cnew - 1][i__ - 1];
                            sm += new_ptr[i__ - 1] * prev_pivot_row_ptr[i__ - 1];
                        }

                        if (sm == 0.0) {continue;}

                        sm *= prev_b;
                        new_ptr[cnew -1] += sm * prev_up;
                        for (int i__ = cnew+1; i__ <= args->n; i__++) {
                            new_ptr[i__ - 1] += sm * prev_pivot_row_ptr[i__ - 1];
                        }
                        
                        atomic_ptrs new_ar = {cold+1, new_ptr};
                        if (ar[lpivot-1+j].compare_exchange_strong(old_ar, new_ar, std::memory_order_seq_cst)){
                            new_ptr = old_ptr;
                        }
                    }
                    else{
                        //std::cout<<"If failure local.iter "<<local.iter<<" cold "<<cold<<'\n';
                        break;
                    }
                }
            }  
	    }

        if (local.start == local.end || local.iter > args->m){
            //pthread_barrier_wait(&barr);
            int next_pivot_update_val = INT_MIN;
            old = gload.load(std::memory_order_seq_cst);
            local.start = local.end = 1;
	        if(local.iter < args->m) {
	            //local.pivot_var = temp_matrix[local.iter][local.iter];
                //double *pivot_row = br[local.iter].load(std::memory_order_seq_cst);
                atomic_ptrs local_ar = ar[local.iter].load(std::memory_order_seq_cst);
                double *pivot_row = local_ar.br;
	            local.pivot_var = pivot_row[local.iter];
                //next_pivot_update_val = pr[local.iter].load(std::memory_order_seq_cst);
                next_pivot_update_val = local_ar.pr;
            }

    	    local.iter += 1;
            if (local.iter - 1 == old.iter && next_pivot_update_val == local.iter - 1){
                  gload.compare_exchange_strong(old, local, std::memory_order_seq_cst);
            }
            //else{
                //std::cout<<"Tid: "<<args->tid<<"If failure local.iter-1 "<<local.iter-1<<" next_pivot_update_val "<<next_pivot_update_val<<'\n';
            //}

            if (local.iter > args->m){
                break;
            }
        }
    }

}

void householder(int num_threads, const std::string& filename){
    double sm = 0, sm1 = 0, cl = 0, clinv = 0, up = 0, b = 0;

    std::cout<<"Num Threads: "<<num_threads<<" Beta: "<<beta<<std::endl;
    std::stringstream savefile;

    std::size_t m, n;
    std::ifstream infile(filename);
    infile >>m;
    infile >>n;
    //std::vector<std::vector<double>> temp_matrix(m, std::vector<double>(n));

    double** temp_matrix = new double*[m];
    for (int i = 0; i<m; i++){
        temp_matrix[i] = new double[n];
    } 
    
    for (std::size_t i = 0; i <m; i++){
        for(std::size_t j = 0; j < n; j++){
            infile >> temp_matrix[i][j];
        }
    }

    pthread_t threads[num_threads];
    args_t thread_args[num_threads];

    std::cout<<"Matrix Size: ("<<m<<","<<n<<")\n";


    gload.store({1,1,1,temp_matrix[0][0]});
    global_ctr.store(0);
    //pivot_val.store(temp_matrix[0][0]);
    
    //pr = new std::atomic<int>[m];
    //br = new std::atomic<double*>[m];

    ar = new std::atomic<atomic_ptrs>[m];

    for (int i=0; i<m; i++){
        //pr[i].store(0, std::memory_order_seq_cst);
        //br[i].store(temp_matrix[i], std::memory_order_seq_cst);
        ar[i].store({0, temp_matrix[i]}, std::memory_order_seq_cst); 
    }

    double *up_ptr = new double[n];
    double *b_ptr = new double[n];

    double tstart = omp_get_wtime();

    for(int i=0; i<num_threads; i++){
        thread_args[i].tid = i;
        thread_args[i].m = m;   thread_args[i].n = n;
        thread_args[i].matrix_ptr = temp_matrix;
        thread_args[i].up_arr = up_ptr;
        thread_args[i].b_arr = b_ptr;
        pthread_create(&threads[i], NULL, thdwork, &thread_args[i]);
    }

    for(int i =0; i<num_threads; i++){
        pthread_join(threads[i], NULL);
    }

    double tend = omp_get_wtime();

    std::cout<<"Finished Householder Transform\n";
    std::cout<<"Time Taken: "<<tend-tstart<<"s\n";

    savefile<<"QR_"<<m<<"x"<<n<<"_barr_helper.txt";

    std::cout<<"Writing output to "<<savefile.str()<<std::endl;

    std::ofstream outfile(savefile.str());

    for (std::size_t i = 0; i < m; i++){
        atomic_ptrs local_ar = ar[i].load(std::memory_order_seq_cst);
        double *temp_row = local_ar.br;
	for (std::size_t j = 0; j < n; j++){
	    //if (j == n -1) outfile << temp_matrix[i][j];
	    if (j == n -1) outfile << temp_row[j];
	    //else outfile<<temp_matrix[i][j]<<' ';
	    else outfile<<temp_row[j]<<' ';
	}
    	outfile<<'\n';
    }

    for(int i = 0; i<m; i++){
        delete[] temp_matrix[i];
    }
    delete[] temp_matrix;
}

int main(int argc, char *argv[]){
    int num_threads = std::stoi(argv[1]);
    std::string filename = argv[2];
    householder(num_threads, filename);
    return 0;
}
