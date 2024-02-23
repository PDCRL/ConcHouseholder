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
    std::vector<std::vector<double>>* matrix_ptr;
};

struct load{
    int start;
    int end;
};

std::atomic<load> gload;

const unsigned int beta = 5;
std::atomic<int> global_ctr;
pthread_barrier_t barr;

std::mutex mtx;

std::vector<double> wtimes(64, 0.0);

void* thdwork(void* params){

    args_t* args = (struct args_t*)params;

    double sm = 0, sm1 = 0, cl = 0, clinv = 0, up = 0, b = 0;
    
    auto temp_matrix = args->matrix_ptr;

    double pivot_var = 0.0;

    int ctr;
    
    pivot_var = (*temp_matrix)[0][0];
    pthread_barrier_wait(&barr);

    for(std::size_t lpivot = 1; lpivot <= args->m; lpivot++){

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

	int old = lpivot-1;
	if(global_ctr.compare_exchange_strong(old, lpivot, std::memory_order_seq_cst)){
		(*temp_matrix)[lpivot-1][lpivot-1] = cl;
	}
        while(1){
            load old, local;
            old = gload.load();
            local = old;

            local.start = old.end;
            local.end = old.end + beta;

            if(local.end > args->m - lpivot){
                local.end = args->m - lpivot +1;
            }

            if(gload.compare_exchange_strong(old, local, std::memory_order_seq_cst)){

                for(std::size_t j = local.start; j < local.end; j++){

                    sm = (*temp_matrix)[lpivot -1 +j][lpivot - 1] * up;
                    
	    	    //std::cout<<"Before: "<<up<<','<<sm<<','<<std::endl;
                    for (std::size_t i__ = lpivot+1; i__ <= args->n; i__++) {
                        sm += (*temp_matrix)[lpivot -1 +j][i__ - 1] * (*temp_matrix)[lpivot - 1][i__ - 1];
                    }

                    if (sm == 0.0) {continue;}

                    sm *= b;

                    (*temp_matrix)[lpivot -1 +j][lpivot -1] += sm * up;

                    for (std::size_t i__ = lpivot+1; i__ <= args->n; i__++) {
                        (*temp_matrix)[lpivot -1 +j][i__ - 1] += sm * (*temp_matrix)[lpivot - 1][i__ - 1];
                    }

                }
            }

            if (local.start == local.end){
                break;
            }
        }
	double tstart = omp_get_wtime();
        pthread_barrier_wait(&barr);
	double tend = omp_get_wtime();
	wtimes[args->tid] += (tend-tstart);
        if (lpivot < args->m){
            pivot_var = (*temp_matrix)[lpivot][lpivot];
        }
	if (args->tid == 0){
	    gload.store({1,1});
	}
        pthread_barrier_wait(&barr);
    }
}

void householder(int num_threads, const std::string& filename){
    double sm = 0, sm1 = 0, cl = 0, clinv = 0, up = 0, b = 0;

    std::cout<<"Num Threads: "<<num_threads<<std::endl;
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

    pthread_t threads[num_threads];
    args_t thread_args[num_threads];

    std::cout<<"Matrix Size: ("<<temp_matrix.size()<<","<<temp_matrix[0].size()<<")\n";

    double tstart = omp_get_wtime();

    double pivot_var = 0.0;

    pthread_barrier_init(&barr, 0, num_threads);
    gload.store({1,1});
    global_ctr.store(0);
    //pivot_val.store(temp_matrix[0][0]);
    for(int i=0; i<num_threads; i++){
        thread_args[i].tid = i;
        thread_args[i].m = m;   thread_args[i].n = n;
        thread_args[i].matrix_ptr = &temp_matrix;
        pthread_create(&threads[i], NULL, thdwork, &thread_args[i]);
    }

    for(int i =0; i<num_threads; i++){
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barr);

    double avg_wait_time = 0.0;
    for (int i = 0; i < num_threads; i++){
	avg_wait_time += wtimes[i];
    } 

    std::cout<< "Avg. wait time in the barrier: "<<avg_wait_time/(num_threads * m)<<'\n';
    double tend = omp_get_wtime();
    std::cout<<"Finished Householder Transform\n";
    std::cout<<"Time Taken: "<<tend-tstart<<"s\n";

    savefile<<"QR_"<<m<<"x"<<n<<"_pthread_np.txt";

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
