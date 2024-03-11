//
// Created by Ali Burak Ünal on 24.03.22.
//

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <chrono>
#include <iomanip>
#include "../../core/rkn.h"
#include "../../utils/parse_options.h"
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cassert>

using namespace std;



int main(int argc, char* argv[]) {
    /*
     * Test the whole prediction process of RKN including the inverse square root of Gram matrix
     */
    if (argc != 14){
        cout << "Calling proxy without specifying Role (1), port (2), address (3), helpers port (4) and helpers adress (5) is not possible." << endl;
        return 1;
    }
    // setup of the proxy and the communication
    uint8_t role = atoi(argv[1]);
    uint16_t cport = atoi(argv[2]);
    string caddress(argv[3]);
    uint16_t hport = atoi(argv[4]);
    string haddress(argv[5]);

    // setup of the parameters of the experiment
    bool random_flag = atoi(argv[6]) != 0; // whether to use random values or a real example
    int n_anc = atoi(argv[7]); // number of anchor points
    int k_mer = atoi(argv[8]); // k-mer length
    double lambda = atof(argv[9]); // adjust the combination of ck[t-1] and ck[t]
    double sigma = atof(argv[10]); // implicitly used in similarity computation
    int run_id = atoi(argv[11]); // the run id of the experiment
    string network = argv[12]; // network type - LAN or WAN

    int s_ind; // test sample index
    int length; // length of the sequence
    // if the random flag is true, the length of the synthetic sequence is set via the parameter. If the random flag
    // is false, then the length of the sequence is set when the sequence is read later in the program. Instead of the
    // length, test sequence id is set when the random flag is false.
    if(random_flag) {
        length = atoi(argv[13]);
    }
    else {
        s_ind = atoi(argv[13]);
    }

    cout << "======================================================" << endl;
    cout << "Parameters: " << endl;
    cout << "Random Flag: " << random_flag << endl;
    cout << "Number of Anchor Points: " << n_anc << endl;
    cout << "k-mer: " << k_mer << endl;
    cout << "Lambda: " << lambda << endl;
    cout << "Sigma: " << sigma << endl;
    cout << "Run ID: " << run_id << endl;
    cout << "Network Type: " << network << endl;
    cout << "Length of sequence (if random_flag is true): " << length << endl;
    cout << "Sequence index (if random_flag is false): " << s_ind << endl;
    cout << "======================================================" << endl;

    //ensure ports are not 0 for helper or client
    cout << "Setting ports for helper/client..." << endl;
    if (cport != 0) {
        assert(cport < 1 << (sizeof(uint16_t) * 8));
    }
    if (hport != 0) {
        assert(hport < 1 << (sizeof(uint16_t) * 8));
    }

    Party *proxy;
    cout << "Creating Proxy..." << endl;
    if (role==0)
        proxy = new Party(proxy1, hport, haddress, cport, caddress);
    else
        proxy = new Party(proxy2, hport, haddress, cport, caddress);


    // setup of the rest of the parameters -- these generally stay the same for the experiments, that's why
    // I did not make them such that they can be set via the arguments of the run
    int n_dim = 20; // number of dimensionality of one-hot encoding
    int n_layer = 1; // number of layers -- so far, we have only one layer
    double reg = 0.1; // I do not remember this?
    double alpha = 1.0 / (pow(sigma, 2) * k_mer);
    string pooling = "gmp"; // pooling -- which is canceled and has no effect
    string tfid = "a.102.1"; // sample id -- a.101.1 - a.102.1 - a.102.4
    string enc = "one_hot"; // encoding type
    string eps = "_eps"; // do not remember?
    double epsilon = 0.01; // epsilon added on top of eigenvalues for numeric problems - to replicate RKN
    bool is_invsqrt_outsourced = true; // true if the inverse square root of Gram matrices of anchor points is outsourced
    bool compute_gt = false; // indicate if the ground truth will be computed
    uint64_t nf_lambda = ConvertToUint64(lambda); // lambda in our number format
    uint64_t nf_alpha = ConvertToUint64(alpha); // lambda in our number format

    uint32_t* params = new uint32_t[4]; // to store the size information for SendBytes calls
    uint64_t** all_x;
    uint64_t*** anchor_points = new uint64_t**[k_mer];
    uint64_t*** tr_anchor_points = new uint64_t**[k_mer]; // transpose of the anchor points in each layer
    uint64_t* weights;
    uint64_t bias;
    uint64_t** invsqrt_of_km_last_layer;
    uint64_t*** gms;
    uint64_t*** kmer_kms;
    uint64_t*** invsqrt_gms;

    string base_fn;
    string folder_name;
    string cecilia_folder = "../";

    auto start = chrono::high_resolution_clock::now();

    if(random_flag) { // random values
        all_x = new uint64_t*[length];
        cout << "Generating data..." << endl;
        for(int s = 0; s < length; s++) {
            all_x[s] = proxy->CreateShare(Random1dData(proxy, (size_t) n_dim, 1.0, false), n_dim);
        }

        // generate a random anchor points
        cout << "Generating anchor points..." << endl;
        for(int i = 0; i < k_mer; i++) {
            anchor_points[i] = proxy->CreateShare(Random2dData(proxy, n_anc, n_dim, 1, false), n_anc, n_dim);

            tr_anchor_points[i] = new uint64_t*[n_dim];
            for(int r = 0; r < n_dim; r++) {
                tr_anchor_points[i][r] = new uint64_t[n_anc];
                for(int c = 0; c < n_anc; c++) {
                    tr_anchor_points[i][r][c] = anchor_points[i][c][r];
                }
            }
        }

        // linear layer for the classification
        weights = proxy->CreateShare(Random1dData(proxy, n_anc + 1, 0.0, 1.0), n_anc + 1);
        bias = weights[n_anc];

        // generate a square matrix as a placeholder for inverse square root matrix
        invsqrt_of_km_last_layer = proxy->CreateShare(Random2dData(proxy, n_anc, n_anc, 1, 10),
                                                      n_anc, n_anc);
    }
    else { // real values
        // prepare the string form of the parameters for the file name
        ostringstream oss;
        oss << setprecision(1) << noshowpoint << lambda;
        std::string str_lmb = oss.str();
        ostringstream oss2;
        oss2 << setprecision(1) << noshowpoint << sigma;
        std::string str_sigma = oss2.str();
        ostringstream oss3;
        oss3 << setprecision(1) << noshowpoint << reg;
        std::string str_reg = oss3.str();

        // sequence
        folder_name = to_string(n_layer) + "_[" + to_string(n_anc) + "]_[" + to_string(k_mer) + "]_[" +
                             str_lmb + "]_[" + str_sigma + "]_" + str_reg;
        base_fn = cecilia_folder + "rkn_results/" +  pooling + "/" + enc + "/" + folder_name + "/" + tfid; // new experiments to validate the correctness
//        cout << "Base folder name: " << base_fn << endl;
        string seq = RecoverSequence(base_fn + "/test_samples.csv", s_ind); // original experiments
        if (seq == "ERROR") {
            cout << "Error in reading the sequence" << endl;
            return -1;
        }
        length = seq.length(); // length of the sequence
//        cout << "Sequence with length " << length << " :" << endl;
        for(char i : seq) {
            cout << i;
        }
        cout << endl;

        all_x = EncodeSequence(proxy, seq);

//        cout << "Reading anchor points..." << endl;
        for(int i = 0; i < k_mer; i++) {
            anchor_points[i] = Read2dArray(proxy, base_fn + "/layer" + to_string(i) + "_k" + to_string(k_mer) +
                                                  "_anc" + to_string(n_anc) + "_dim" + to_string(n_dim), n_anc, n_dim);

            tr_anchor_points[i] = new uint64_t*[n_dim];
            for(int r = 0; r < n_dim; r++) {
                tr_anchor_points[i][r] = new uint64_t[n_anc];
                for(int c = 0; c < n_anc; c++) {
                    tr_anchor_points[i][r][c] = anchor_points[i][c][r];
                }
            }
        }

        // linear layer for the classification
        weights = Read1dArray(proxy, base_fn + "/linear_layer_k" + to_string(k_mer) + "_anc" + to_string(n_anc) +
                                     "_dim" + to_string(n_dim), n_anc + 1);
//        cout << "Weights are read" << endl;
        bias = weights[n_anc];

        // inverse square root of Gram matrices of anchor points if is_invsqrt_outsourced is true
        if(is_invsqrt_outsourced) {
            invsqrt_of_km_last_layer = Read2dArray(proxy, base_fn + "/invsqrt_layer" + to_string(k_mer - 1), n_anc, n_anc);
            if(invsqrt_of_km_last_layer == nullptr) {
                cout << "Inverse square root matrix cannot be read" << endl;
                return -1;
            }
        }
    }

//    cout << "Preparation is done!" << endl;

    int size = k_mer * n_anc * n_dim;
    int size2 = k_mer * n_anc;

    // generate a random data to represent the output of the previous time point at the same layer
    uint64_t* ct = Zero1dData(proxy, size2 + n_anc);
    uint64_t* initial_ct = Zero1dData(proxy, size2 + n_anc);
    for(int i = 0; i < n_anc; i++) {
        ct[i] = proxy->GetPRole() * ((uint64_t) 1 << FRACTIONAL_BITS);
        initial_ct[i] = ct[i];
    }

    auto start_initial_mapping = chrono::high_resolution_clock::now();

    // generate random sequence data
    for(int s = 0; s < length; s++) {
        // b part of the ppRKN
        uint64_t* str_z = new uint64_t[size];

        for(int i = 0; i < k_mer; i++) {
            for(int j = 0; j < n_anc; j++) {
                for(int k = 0; k < n_dim; k++) {
                    str_z[(i * n_anc * n_dim) + (j * n_dim) + k] = anchor_points[i][j][k];
                }
            }
        }
//        cout << "iteration " << s << endl;
        params[0] = size;
        params[1] = size2;
        proxy->SendBytes(rknIteration, params, 2);
        uint64_t* tmp_ct = RknIteration(proxy, all_x[s], str_z, ct, n_dim, n_anc, k_mer, nf_lambda, nf_alpha);
        copy(tmp_ct, tmp_ct + size2, ct + n_anc);
        delete [] str_z;
    }
    cout << "Initial mapping is done" << endl;

    auto end_initial_mapping = chrono::high_resolution_clock::now();
    // convert c[t] to matrix
    uint64_t** mat_ct = new uint64_t *[k_mer];
    for(int i = 0; i < k_mer; i++) {
        mat_ct[i] = new uint64_t[n_anc];
        for(int j = 0; j < n_anc; j++) {
            mat_ct[i][j] = ct[n_anc + (i * n_anc) + j];
        }
    }

//    cout << "mat_ct is filled up" << endl;

    // In case we compute the inverse square root of Gram matrix of anchor points from scratch
    if (!is_invsqrt_outsourced) {
        // Gram matrices of the anchor points
        params[0] = k_mer;
        params[1] = n_anc;
        params[2] = n_dim;
        params[3] = n_anc;
        proxy->SendBytes(coreVectorisedMatrixMatrixMultiply, params, 4);
        gms = MatrixMatrixMultiply(proxy, anchor_points, tr_anchor_points, k_mer, n_anc, n_dim, n_anc);
        cout << "Gram matrix of the anchor points are computed" << endl;

        params[0] = k_mer;
        params[1] = n_anc;
        proxy->SendBytes(rknGaussianKernel, params, 2);
        kmer_kms = GaussianKernel(proxy, gms, ConvertToUint64(alpha), k_mer, n_anc);
        cout << "Kernel matrix of the anchor points are computed using alpha " << alpha << endl;

        // inverse square root of the Gram matrices
        params[0] = k_mer;
        params[1] = n_anc;
        proxy->SendBytes(rknVectorisedInverseSqrt, params, 2);
        invsqrt_gms = InverseSqrt(proxy, kmer_kms, k_mer, n_anc, epsilon);
        cout << "Inverse square of the kernel matrices are computed" << endl;

        invsqrt_of_km_last_layer = invsqrt_gms[k_mer - 1];
    }
    auto end_invsqrt = chrono::high_resolution_clock::now();

    // final mapping of the sequence
    params[0] = n_anc;
    params[1] = n_anc;
    proxy->SendBytes(coreMatrixVectorMultiply, params, 2);
    uint64_t* x_mapping = MatrixVectorMultiply(proxy, invsqrt_of_km_last_layer, mat_ct[k_mer - 1],
                                               n_anc, n_anc); // mapping of only the last layer

    cout << "Final mapping of the sample is computed" << endl;

    // linear classifier layer
    params[0] = n_anc;
    proxy->SendBytes(coreDotProduct, params, 1);
    uint64_t prediction = DotProduct(proxy, weights, x_mapping, n_anc) + bias;
    cout << "Prediction is obtained" << endl;

    proxy->SendBytes(coreEnd);
    auto end = chrono::high_resolution_clock::now();
    double time_taken;
    double exe_times[5];

    time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count() * 1e-9;
    cout<<"Total_Time: " << fixed << time_taken << setprecision(9) << " sec" << endl;
    exe_times[0] = time_taken;

    time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start_initial_mapping).count() * 1e-9;
    cout<<"Process_Without_Preparation_Time: " << fixed << time_taken << setprecision(9) << " sec" << endl;
    exe_times[1] = time_taken;

    time_taken = chrono::duration_cast<chrono::nanoseconds>(end_initial_mapping - start_initial_mapping).count() * 1e-9;
    cout<<"Initial_Mapping_Time: " << fixed << time_taken << setprecision(9) << " sec" << endl;
    exe_times[2] = time_taken;

    if (!is_invsqrt_outsourced) {
        time_taken = chrono::duration_cast<chrono::nanoseconds>(end_invsqrt - end_initial_mapping).count() * 1e-9;
        cout<<"INVSQRT_Time: " << fixed << time_taken << setprecision(9) << " sec" << endl;
        exe_times[3] = time_taken;
    }

    time_taken = chrono::duration_cast<chrono::nanoseconds>(end - end_invsqrt).count() * 1e-9;
    cout<<"Linear_Classifier_Time: " << fixed << time_taken << setprecision(9) << " sec" << endl;
    exe_times[4] = time_taken;


    double d_prediction = ConvertToDouble(Reconstruct(proxy, prediction));
    PrintValue("Prediction", d_prediction);

    // writing the execution time results into a file
    string result_fn = cecilia_folder + "exp_runners/rkn_experiments/pprkn_inference_results/";
    if(random_flag) {
        result_fn = result_fn + "synthetic/p" +
                    to_string(proxy->GetPRole()) + "_" + to_string(n_anc) + "_" + to_string(k_mer) + "_" +
                    to_string(length) + "_" + to_string(run_id) + "_" + network + ".csv";
    }
    else {
        result_fn = result_fn + "real/p" +
                    to_string(proxy->GetPRole()) + "_" + to_string(n_anc) + "_" + to_string(k_mer) + "_" +
                    to_string(run_id) + "_" + network + "_" + tfid + "_" + to_string(s_ind) + "_" +
                    to_string(length) + ".csv";
    }

    ofstream res_file (result_fn);
    if (res_file.is_open())
    {
        for(int i = 0; i < 4; i++) {
            res_file << exe_times[i] << ",";
        }
        res_file << exe_times[4];
        if(!random_flag) {
            res_file << "," << d_prediction;
        }
        res_file << "\n";
        res_file.close();
    }
    else {
        cout << "Unable to open file: " << result_fn << endl;
        return -1;
    }
    // end of writing the execution time results into a file



    for(int i = 0; i < k_mer; i++) {
        delete [] mat_ct[i];
    }
    delete [] mat_ct;


    if(!is_invsqrt_outsourced) {
        for(int i = 0; i < k_mer; i++) {
            for(int j = 0; j < n_anc; j++) {
                delete [] gms[i][j];
                delete [] invsqrt_gms[i][j];
            }
            delete [] gms[i];
            delete [] invsqrt_gms[i];
        }
        delete [] gms;
        delete [] invsqrt_gms;
    }


    // ********************************************************************
    // ********************************************************************
    // ********************************************************************

//    if(compute_gt) {
//        // Ground truth computation
//        cout << "Ground truth computation starts..." << endl;
//        double*** rec_anc_points = new double**[k_mer];
//        for(int i = 0; i < k_mer; i++) {
//            rec_anc_points[i] = ConvertToDouble(Reconstruct(proxy, anchor_points[i], n_anc, n_dim), n_anc, n_dim);
//        }
//        cout << "check 1" << endl;
//        double** rec_all_x = ConvertToDouble(Reconstruct(proxy, all_x, length, n_dim), length, n_dim);
//        double* rec_ct = ConvertToDouble(Reconstruct(proxy, initial_ct, size2 + n_anc), size2 + n_anc);
//        cout << "check 2" << endl;
//        double** gt_dp = new double*[k_mer];
//        double** exp_gt_dp = new double*[k_mer];
//        double** gt_skt = new double*[k_mer];
//        double** gt_ckt = new double*[k_mer];
//        for(int k = 0; k < k_mer; k++) {
//            gt_dp[k] = new double[n_anc];
//            exp_gt_dp[k] = new double[n_anc];
//            gt_skt[k] = new double[n_anc];
//            gt_ckt[k] = new double[n_anc];
//        }
//        cout << "check 3" << endl;
//        for(int iter = 0; iter < length; iter++) {
//            cout << "Iteration " << iter << endl;
//            // Ground truth: b part
//            // dot product
//            for(int k = 0; k < k_mer; k++) {
//                for(int i = 0; i < n_anc; i++) {
//                    double tmp_sum = 0;
//                    for(int j = 0; j < n_dim; j++) {
//                        tmp_sum += rec_all_x[iter][j] * rec_anc_points[k][i][j];
//                    }
//                    gt_dp[k][i] = tmp_sum;
//                    exp_gt_dp[k][i] = exp(alpha * (tmp_sum - 1));
//                }
//            }
//
//            // Ground truth: c_{k-1}[t-1] * b_{l}[t]
//            for(int i = 0; i < k_mer; i++) {
//                for(int j = 0; j < n_anc; j++) {
//                    gt_skt[i][j] = exp_gt_dp[i][j] * rec_ct[i * n_anc + j];
//                }
//            }
//
//            // Ground truth: lambda * c_{k}[t-1] + s_{k}[t]
//            for(int i = 0; i < k_mer; i++) {
//                for(int j = 0; j < n_anc; j++) {
//                    gt_ckt[i][j] = lambda * gt_skt[i][j] + (1 - lambda) * rec_ct[(i + 1) * n_anc + j];
//                }
//            }
//
//            // update c[t] based on the result of the mappings in each k-mer
//            for(int i = 1; i < k_mer + 1; i++) {
//                for(int j = 0; j < n_anc; j++) {
//                    rec_ct[i * n_anc + j] = gt_ckt[i - 1][j];
//                }
//            }
//        }
//        cout << "check 4" << endl;
//        // delete dynamically allocated arrays
//        for(int i = 0; i < k_mer; i++) {
//            delete [] gt_dp[i];
//            delete [] exp_gt_dp[i];
//            delete [] gt_skt[i];
//            delete [] gt_ckt[i];
//        }
//        delete [] gt_dp;
//        delete [] exp_gt_dp;
//        delete [] gt_skt;
//        delete [] gt_ckt;
//
//        // ----------------------------------------------------------------------------------------------
//        // generate Gram matrices
//        double*** gt_gms = new double**[k_mer];
//        gt_gms[0] = InplaceDotProduct(rec_anc_points[0], rec_anc_points[0], n_anc, n_dim);
//        for(int j = 0; j < n_anc; j++) {
//            for(int k = j; k < n_anc; k++) {
//                gt_gms[0][j][k] = exp(alpha * (gt_gms[0][j][k] - 1));
//                gt_gms[0][k][j] = gt_gms[0][j][k];
//            }
//        }
//
//        // initialize the rest of the gt_gms array
//        for(int i = 1; i < k_mer; i++) {
//            gt_gms[i] = new double*[n_anc];
//            for(int j = 0; j < n_anc; j++) {
//                gt_gms[i][j] = new double[n_anc];
//            }
//        }
//
//        //    proxy->Print2dArray("GT Gram matrix 0", gt_gms[0], n_anc, n_anc);
//        for(int i = 1; i < k_mer; i++) {
//            double** tmp_gt_gms = InplaceDotProduct(rec_anc_points[i], rec_anc_points[i], n_anc, n_dim);
//            for(int j = 0; j < n_anc; j++) {
//                for(int k = j; k < n_anc; k++) {
//                    gt_gms[i][j][k] = exp(alpha * (tmp_gt_gms[j][k] - 1)) * gt_gms[i - 1][j][k];
//                    gt_gms[i][k][j] = gt_gms[i][j][k];
//                }
//            }
//        }
//
//        // compute kernel matrices
//        double*** gt_kms = new double**[k_mer];
//        for(int i = 0; i < k_mer; i++) {
//            gt_kms[i] = new double*[n_anc];
//            for(int j = 0; j < n_anc; j++) {
//                gt_kms[i][j] = new double[n_anc];
//                for(int k = 0; k < n_anc; k++) {
//                    gt_kms[i][j][k] = gt_gms[i][j][k];
//                }
//            }
//        }
//
//        double*** rec_kmer_kms = new double**[k_mer];
//        for(int g = 0; g < k_mer; g++) {
//            rec_kmer_kms[g] = ConvertToDouble(Reconstruct(proxy, kmer_kms[g], n_anc, n_anc), n_anc, n_anc);
//        }
//
//        cout << "check 5" << endl;
//        double** gt_res = new double*[k_mer];
//        double** gt_eigvals = new double*[k_mer];
//        double** AT_gt_eigvals = new double*[k_mer];
//        for(int g = 0; g < k_mer; g++) {
//            double* straighten_G = new double[n_anc * n_anc];
//            double* AT_straighten_G = new double[n_anc * n_anc];
//            for(uint32_t i = 0; i < n_anc * n_anc; i++) {
//                straighten_G[i] = gt_kms[g][i % n_anc][i / n_anc];
//                AT_straighten_G[i] = rec_kmer_kms[g][i % n_anc][i / n_anc];
//            }
//
//            // ****************************************************************************************************
//            EigenSolver<Matrix<double, Dynamic, Dynamic, RowMajor>> AT_ges;
//            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> AT_matrix_G(AT_straighten_G, n_anc, n_anc);
//            AT_ges.compute(AT_matrix_G);
//            Matrix<double, Dynamic, 1> AT_eig_vals = AT_ges.eigenvalues().real();
//            AT_gt_eigvals[g] = new double[n_anc];
//            Map<Matrix<double, Dynamic, 1>>(AT_gt_eigvals[g], n_anc) = AT_eig_vals;
//            // ****************************************************************************************************
//
//            EigenSolver<Matrix<double, Dynamic, Dynamic, RowMajor>> ges;
//            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> matrix_G(straighten_G, n_anc, n_anc);
//            ges.compute(matrix_G);
//            Matrix<double, Dynamic, Dynamic, RowMajor> eig_vecs = ges.eigenvectors().real();
//            Matrix<double, Dynamic, 1> eig_vals = ges.eigenvalues().real();
//
//            gt_eigvals[g] = new double[n_anc];
//            Map<Matrix<double, Dynamic, 1>>(gt_eigvals[g], n_anc) = eig_vals;
//
//            Matrix<double, Dynamic, Dynamic, RowMajor> vals = eig_vals;
//            double* tmp_str_invsqrt = new double[n_anc * n_anc];
//            Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(tmp_str_invsqrt, n_anc, n_anc) =
//                    eig_vecs * (vals.cwiseSqrt().array() + epsilon).matrix().cwiseInverse().asDiagonal() * Transpose(eig_vecs);
//            double** tmp_invsqrt_gm = new double*[n_anc];
//            for(int at = 0; at < n_anc; at++) {
//                tmp_invsqrt_gm[at] = new double[n_anc];
//                for(int kafa = 0; kafa < n_anc; kafa++) {
//                    tmp_invsqrt_gm[at][kafa] = tmp_str_invsqrt[at * n_anc + kafa];
//                }
//            }
//
//            gt_res[g] = MultiplyMatrixVector(tmp_invsqrt_gm, &rec_ct[(g + 1) * n_anc], n_anc, n_anc);
//
//            // deleting dynamically allocated arrays
//            delete [] straighten_G;
//            delete [] tmp_str_invsqrt;
//            for(int d = 0; d < n_anc; d++) {
//                delete [] tmp_invsqrt_gm[d];
//            }
//            delete [] tmp_invsqrt_gm;
//        }
//        cout << "check 6" << endl;
//
//        double* rec_weights = ConvertToDouble(Reconstruct(proxy, weights, n_anc + 1), n_anc + 1);
//        double rec_bias = rec_weights[n_anc];
//        double gt_prediction = MultiplyVectorVector(gt_res[k_mer - 1], rec_weights, n_anc) + rec_bias;
//        cout << "check 7" << endl;
//        double* total_diff = new double[k_mer];
//        for(int i = 0; i < k_mer; i++) {
//            total_diff[i] = 0;
//        }
//        cout << "check 8" << endl;
//        double** rec_x_mapping = ConvertToDouble(Reconstruct(proxy, x_mapping, k_mer, n_anc), k_mer, n_anc);
//        cout << "rec_x_mapping is done" << endl;
//        double **diff = new double*[n_anc];
//        for(int i = 0; i < n_anc; i++) {
//            cout << "Anchor point " << i << endl;
//            diff[i] = new double[k_mer];
//            for(int j = 0; j < k_mer; j++) {
//                cout << "k-mer " << j << endl;
//                diff[i][j] = gt_res[j][i] - rec_x_mapping[j][i];
//                total_diff[j] += abs(diff[i][j]);
//            }
//        }
//        PrintValue("GT Prediction", gt_prediction);
//        PrintValue("|Prediction - GT Prediction|", abs(ConvertToDouble(Reconstruct(proxy, prediction)) - gt_prediction));
//
//        // delete the dynamically allocated arrays
//        for(int i = 0; i < k_mer; i++) {
//            for(int j = 0; j < n_anc; j++) {
//                delete [] rec_anc_points[i][j];
//                delete [] gt_gms[i][j];
//                delete [] gt_kms[i][j];
//            }
//            delete [] rec_anc_points[i];
//            delete [] gt_gms[i];
//            delete [] gt_kms[i];
//        }
//        delete [] rec_anc_points;
//        delete [] gt_gms;
//        delete [] gt_kms;
//        delete [] rec_ct;
//
//        for(int i = 0; i < length; i++) {
//            delete [] rec_all_x[i];
//        }
//        delete [] rec_all_x;
//
//        for(int i = 0; i < n_anc; i++) {
//            delete [] diff[i];
//        }
//        delete [] diff;
//    }
//
//    // delete the rest of the dynamically allocated arrays
//    for(int i = 0; i < k_mer; i++) {
//        delete [] x_mapping[i];
//        for(int j = 0; j < n_anc; j++) {
//            delete [] kmer_kms[i][j];
//            delete [] anchor_points[i][j];
//        }
//        delete [] kmer_kms[i];
//        delete [] anchor_points[i];
//    }
//    delete [] x_mapping;
//    delete [] kmer_kms;
//    delete [] anchor_points;
//    for(int i = 0; i < length; i++) {
//        delete [] all_x[i];
//    }
//    delete [] all_x;
}

