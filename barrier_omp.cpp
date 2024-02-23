#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <omp.h>

using vvd = std::vector<std::vector<double>>;
const int beta = 10;

void compute_row_update(vvd& mat, int m, int n, int lpivot, int j, double up, double b){
    
    double sm = mat[lpivot-1 +j][lpivot-1] * up;
    
    for (int i__ = lpivot+1; i__ <= n; i__++){
        sm += mat[lpivot-1 +j][i__-1] * mat[lpivot-1][i__-1];
    }
    if (sm == 0.0 ) return;

    sm *= b;
    
    mat[lpivot-1 +j][lpivot-1] += sm * up;
    
    for (int i__ = lpivot+1; i__ <= n; i__++){
        mat[lpivot-1 +j][i__-1] += sm * mat[lpivot-1][i__-1];
    }
}

void compute_pivot_update(vvd& mat, int m, int n, int lpivot, double& up, double& b){
    double sm = 0.0, sm1 = 0.0;
    double cl = std::fabs(mat[lpivot-1][lpivot-1]);
    
    for (int k = lpivot+1; k <= n; k++){
        sm = std::fabs(mat[lpivot-1][k-1]);
        sm1 += sm * sm;
        cl = std::max(mat[lpivot-1][k-1], cl);
    }
    double clinv = 1.0/cl;
    double d__1 = mat[lpivot-1][lpivot-1] * clinv;
    sm = d__1 * d__1;
    sm += sm1 * clinv * clinv;
    cl *= std::sqrt(sm);

    if (mat[lpivot-1][lpivot-1] > 0.0) cl = -cl;

    up = mat[lpivot-1][lpivot-1] - cl;
    mat[lpivot-1][lpivot-1] = cl;

    b = up * mat[lpivot-1][lpivot-1];

    b = 1.0/b;
}

void compute_transform(vvd& mat, int m, int n, int num_threads){
    double up, b;
    
    for (int lpivot = 1; lpivot <= m; lpivot++){
        
        compute_pivot_update(mat, m, n, lpivot, up, b);
        
        #pragma omp parallel for num_threads(num_threads) schedule(guided, beta)
        for (int j = 1; j <= m - lpivot; j++){
            compute_row_update(mat, m, n, lpivot, j, up, b);
        }
    }
}

void householder(int num_threads, const std::string& filename){

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

    double tstart = omp_get_wtime();
    compute_transform(temp_matrix, m, n, num_threads); 
    double tend = omp_get_wtime();

    std::cout<<"Finished Householder Transform\n";
    std::cout<<"Time Taken: "<<tend-tstart<<"s\n";

    savefile<<"QR_"<<m<<"x"<<n<<"_tasks.txt";

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
