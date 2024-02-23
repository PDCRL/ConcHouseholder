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

struct args_t{
    int tid;
    std::size_t m;
    std::size_t n;
    double *pivots;
    std::vector<std::vector<double>>* matrix_ptr;
};

struct load{
    unsigned int iter;
    unsigned int start;
    unsigned int end;
    double pivot_var;
};

std::atomic<load> gload;

const unsigned int beta = 10;
std::atomic<int> global_ctr;

//std::atomic<std::atomic<int* >* > prev_pr;
std::atomic<std::atomic<int>* > pr;
int* p;
pthread_barrier_t barr;

std::mutex mtx;

void* thdwork(void* params){

    args_t* args = (struct args_t*)params;

    double sm = 0, sm1 = 0, cl = 0, clinv = 0, up = 0, b = 0, pivot_var = 0.0;
    
    
    auto temp_matrix = args->matrix_ptr;

    int ctr = 0;
    
    while(1){
        load old, local;
        old = gload.load(std::memory_order_relaxed);
        local = old;

        local.start = old.end;
        local.end = old.end + beta;

        if(local.end > args->m - local.iter){
            local.end = args->m - local.iter +1;
            /*
	    if(local.iter < args->m) {
	        //local.pivot_var = (*temp_matrix)[local.iter][local.iter];
            	//local.end = args->m - local.iter +1;
	    }
	    else {
		local.end = 1;
	    }
            */
	}

        if(local.iter <= args->m && gload.compare_exchange_strong(old, local, std::memory_order_seq_cst)){
            int lpivot = local.iter;

            if (ctr != lpivot){
                ctr = lpivot;
		        pivot_var = local.pivot_var;
                //std::cout<<"lpivot "<<lpivot<<" tid "<<args->tid<<" local pivot var "<<pivot_var<<'\n';
		        cl = std::fabs(pivot_var);
		        sm1 = 0.0;

		        for(std::size_t k = lpivot+1; k <= args->n; k++){
		            sm = std::fabs((*temp_matrix)[lpivot-1][k-1]);
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
		        //if(global_ctr.compare_exchange_strong(old_ctr, lpivot, std::memory_order_seq_cst)){
                    args->pivots[lpivot-1] = cl;
			        //(*temp_matrix)[lpivot-1][lpivot-1] = cl;
		        //}
            }
            for(std::size_t j = local.start; j < local.end; j++){
                while (1){
                    int cold = pr[lpivot-1+j].load(std::memory_order_seq_cst);

                    if (cold == lpivot - 1){
	    	        sm = (*temp_matrix)[lpivot -1 +j][lpivot - 1] * up;
	    
	                for (std::size_t i__ = lpivot+1; i__ <= args->n; i__++) {
		            sm += (*temp_matrix)[lpivot -1 +j][i__ - 1] * (*temp_matrix)[lpivot - 1][i__ - 1];
	                }

	                if (sm == 0.0) {continue;}

	                sm *= b;
	                (*temp_matrix)[lpivot -1 +j][lpivot -1] += sm * up;

	                for (std::size_t i__ = lpivot+1; i__ <= args->n; i__++) {
		            (*temp_matrix)[lpivot -1 +j][i__ - 1] += sm * (*temp_matrix)[lpivot - 1][i__ - 1];
	                }
                        pr[lpivot-1+j].store(cold+1, std::memory_order_seq_cst);
                        break;
                     }
                //mtx.lock();
                //p[lpivot-1+j]++;
                //mtx.unlock();
                //while (1){
		    //int cold = pr[lpivot-1+j].load(std::memory_order_seq_cst);
                    //int cnew = cold+1;
                    //if (pr[lpivot-1+j].compare_exchange_strong(cold, cnew)){
                        //break;
                    //}
                }
	    }
        }

        if (local.start == local.end || local.iter > args->m){
            //pthread_barrier_wait(&barr);
            int update_val = -10000;
            old = gload.load();
            local.start = local.end = 1;
	    if(local.iter < args->m) {
	        local.pivot_var = (*temp_matrix)[local.iter][local.iter];
                update_val = pr[local.iter].load(std::memory_order_seq_cst);
            }

            //mtx.lock();
            //int update_val = *(pr[local.iter].load());
	        //for (int i = 0; i < args->m; i++){
            //    std::cout<<p[i]<<' ';
            //}
            //std::cout<<'\n';
            //mtx.unlock();


	    local.iter += 1;
            if (local.iter - 1 == old.iter && update_val == local.iter - 1){
                gload.compare_exchange_strong(old, local, std::memory_order_seq_cst);
            }
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
    std::vector<std::vector<double>> temp_matrix(m, std::vector<double>(n));
    
    for (std::size_t i = 0; i <m; i++){
        for(std::size_t j = 0; j < n; j++){
            infile >> temp_matrix[i][j];
        }
    }

    double *pivots = new double[m];

    pthread_t threads[num_threads];
    args_t thread_args[num_threads];

    std::cout<<"Matrix Size: ("<<temp_matrix.size()<<","<<temp_matrix[0].size()<<")\n";

    double tstart = omp_get_wtime();

    pthread_barrier_init(&barr, 0, num_threads);
    gload.store({1,1,1,temp_matrix[0][0]});
    global_ctr.store(0);
    //pivot_val.store(temp_matrix[0][0]);
    
    pr = new std::atomic<int>[m];
    for (int i=0; i<m; i++){
        pr[i].store(0, std::memory_order_seq_cst);
    }
    
    /*
    p = new int[m];
    for (int i = 0; i<m; i++){
        p[i] = 0;
    }
    */
    for(int i=0; i<num_threads; i++){
        thread_args[i].tid = i;
        thread_args[i].m = m;   thread_args[i].n = n;
        thread_args[i].pivots = pivots;
        thread_args[i].matrix_ptr = &temp_matrix;
        pthread_create(&threads[i], NULL, thdwork, &thread_args[i]);
    }

    for(int i =0; i<num_threads; i++){
        pthread_join(threads[i], NULL);
    }

    double tend = omp_get_wtime();
    for(int i = 0; i<m; i++)
        temp_matrix[i][i] = pivots[i];

    pthread_barrier_destroy(&barr);

    std::cout<<"Finished Householder Transform\n";
    std::cout<<"Time Taken: "<<tend-tstart<<"s\n";


    savefile<<"QR_"<<m<<"x"<<n<<"_barrier_free.txt";

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
