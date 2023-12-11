//
// Created by Debora Jutz on 03.02.22.
//

#ifndef CECILIA_TEST_FUNCTIONS_H
#define CECILIA_TEST_FUNCTIONS_H

#include "flib.h"
#include "../core/core.h"
#include <fstream>
#include <tuple>
#include <random>

#include <sstream>
// Random matrix generation functions
 /**
  *
  * @param proxy
  * @param size size of the vector to be created
  * @param max_num highest random number to be contained in the returned data, default is 10
  * @param neg_flag allow negative values as default, set to false if they shall not be allowed
  * @return
  */
 static double* Random1dData(Party *proxy, size_t size, double max_num= 10, bool neg_flag= true) {
     double* mat_data = new double[size];
     for (int i = 0; i < size; i++) {
         mat_data[i] = max_num * (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
         if (neg_flag && rand() % 2 == 0) {
             mat_data[i] *= -1;
         }
     }
     return mat_data;
 }

 static double* Random1dData(Party *proxy, int size, double min_num, double max_num) {
    double* mat_data = new double[size];
    random_device rd; // obtain a random number from hardware
    mt19937 gen(rd()); // seed the generator
    uniform_real_distribution<> distr(min_num, max_num); // define the range
    for (int i = 0; i < size; i++) {
        mat_data[i] = distr(gen);
    }
    return mat_data;
}

static double** Random2dData(Party *proxy, size_t n_row, size_t n_col, double min_num, double max_num) {
    double d_tmp1;
    uint64_t tmp1;
    double** mat_data = new double *[n_row];
    random_device rd; // obtain a random number from hardware
    mt19937 gen(rd()); // seed the generator
    uniform_real_distribution<> distr(min_num, max_num); // define the range
    for (size_t i = 0; i < n_row; i++) {
        mat_data[i] = new double[n_col];
        for(size_t j = 0; j < n_col; j++) {
            mat_data[i][j] = distr(gen);
        }
    }
    return mat_data;
}

static uint64_t** RandomGramMatrix(Party *proxy, int n_row, int n_col) {
    double d_tmp1;
    uint64_t tmp1, a;
    int role = proxy->GetPRole();

    // data matrix generation
    double **rand_matrix = new double *[n_row];
    for(int i = 0; i < n_row; i++) {
        rand_matrix[i] = new double[n_col];

        double tmp_sum = 0;
        for(int j = 0; j < n_col; j++) {
            d_tmp1 = 10 * (static_cast <float> (proxy->GenerateRandom()) / static_cast <float> (RAND_MAX));
            tmp_sum += pow(d_tmp1, 2);
            rand_matrix[i][j] = d_tmp1;
        }
    }

    // gram matrix computation
    double** gram_matrix = new double*[n_row];
    for(int i = 0; i < n_row; i++) {
        gram_matrix[i] = new double[n_row];
        for(int j = 0; j < n_row; j++) {
            double tmp_sum = 0;
            for(int k = 0; k < n_col; k++) {
                tmp_sum += rand_matrix[i][k] * rand_matrix[j][k];
            }
            gram_matrix[i][j] = tmp_sum;
        }
    }

    // share generation
    uint64_t** invsqrt_data = proxy->CreateShare(gram_matrix, n_row, n_row);
    delete[] gram_matrix;
    return invsqrt_data;
}

/**
 * Generate a random matrix whose values are stored in a simple vector. Matrix values as well as size are random but
 * some constraints exist: (1) matrix size will be high enough so that at least 4 windows fit in total;
 *                         (2) window size and matrix size match, meaning that
 *                         matrix row size = x * window row size and matrix column size = y * window column size
 *                         (3) row and column size will be greater 1.
 * @param proxy
 * @param matrix_number specifies the number of matrices of the random shape that shall be generated.
 *                      default is 2, so the matrix and a tmp vector are provided to generated secret shares.
 * @param max_num maximum value to be allowed in the matrix
 * @param neg_flag specifies if negative values are allowed.
 * @return a tuple containing two pointers:
 *          pointer of type uint64_t -> pointing to the generated matrix data
 *          pointer of type uint32_t -> pointing to four values in the given order:
 *                  matrix column size
 *                  matrix row size
 *                  window column size
 *                  window row size
 *         (matrix column size * matrix row size) will be the dimension of the matrix (m_size) and the generated matrix data will contain (m_size * matrix_number) values
 */
static double* RandomWindowMatrix(Party *proxy, uint8_t matrix_number= 2, double max_num= 255.99, bool neg_flag= true){
    cout << "Generate matrix with random values... " << endl;
    uint32_t mColSize = 0, mRowSize = 0;
    uint32_t w_dim = 0; // w_rows = 0;
    while (w_dim < 1){
        w_dim = proxy->GenerateCommonRandom() % (L_BIT / 4);  // divide by 2 so that a valid mColSize can be found
    }
    //TODO adapt to asymmetric window sizes
    /**
    while (w_rows < 1){
        w_rows = proxy->GenerateCommonRandom() % (L/4);
    }*/
    while (mColSize <= w_dim || (mColSize % w_dim) != 0){ // matrix contains at least 2 windows, windows fit completely
        mColSize = proxy->GenerateCommonRandom() % L_BIT;
    }
    while (mRowSize <= w_dim || (mRowSize % w_dim) != 0){
        mRowSize = proxy->GenerateCommonRandom() % L_BIT;
    }
    uint64_t mSize = mRowSize * mColSize;
    //                            for the 4 random parts: matrix A and B, mTmp A and B
    double *randData = new double [mSize*matrix_number + 4]; // +4 for the 4 values specifying matrix + window dimensions
    randData = Random1dData(proxy, mSize * matrix_number, max_num, neg_flag);
    randData[-4] = mColSize;
    randData[-3] = mRowSize;
    randData[-2] = w_dim;
    randData[-1] = 0;
    return randData;
}


/*
 * Printing functions for debugging purposes. There are functions to print out secret shared scalar, 1D and
 * 2D arrays either in secret shared form or after reconstruction. There are similar functions for plaintext data as well.
 * Moreover, for 1D and 2D arrays, there is an option that one can set to print out horizontally or vertically.
 */
// Printing the data of type uint64_t for debugging purposes
void Print2dArray(string const &str1, uint64_t** x, uint32_t n_row, uint32_t n_col, bool horizontal= true) {
    // horizontal: true if the resulting print-out is desired row-by-col
    cout << "======================= " << str1 << " =======================" << endl;
    if(horizontal) {
        for(uint32_t i = 0; i < n_row; i++) {
            cout << i << ".\t";
            for(uint32_t j = 0; j < n_col; j++) {
                cout << x[i][j] << "\t";
            }
            cout << endl;
        }
    }
    else {
        for(uint32_t i = 0; i < n_col; i++) {
            cout << i << ".\t";
            for(uint32_t j = 0; j < n_row; j++) {
                cout << x[j][i] << "\t";
            }
            cout << endl;
        }
    }
    cout << "==============================================================" << endl;
}

void Print1dArray(string const &str1, uint64_t* x, uint32_t size, bool horizontal= true) {
    cout << "======================= " << str1 << " =======================" << endl;
    if(horizontal) {
        for (uint32_t i = 0; i < size; i++) {
            cout << i << ". " << x[i] << "\t";
        }
        cout << endl;
    }
    else {
        for(uint32_t i = 0; i < size; i++) {
            cout << i << ". " << x[i] << endl;
        }
    }
    cout << "==============================================================" << endl;
}

void PrintValue(string const &str1, uint64_t x) {
    cout << "======================= " << str1 << " =======================" << endl;
    cout << x << endl;
    cout << "==============================================================" << endl;
}

// Printing the data of type double for debugging purposes
void Print2dArray(string const &str1, double** x, uint32_t n_row, uint32_t n_col, bool horizontal= true) {
    // horizontal: true if the resulting print-out is desired row-by-col
    cout << "======================= " << str1 << " =======================" << endl;
    if(horizontal) {
        for(uint32_t i = 0; i < n_row; i++) {
            cout << i << ".\t";
            for(uint32_t j = 0; j < n_col; j++) {
                cout << x[i][j] << "\t";
            }
            cout << endl;
        }
    }
    else {
        for(uint32_t i = 0; i < n_col; i++) {
            cout << i << ".\t";
            for(uint32_t j = 0; j < n_row; j++) {
                cout << x[j][i] << "\t";
            }
            cout << endl;
        }
    }
    cout << "==============================================================" << endl;
}

void Print1dArray(string const &str1, double* x, uint32_t size, bool horizontal= true) {
    cout << "======================= " << str1 << " =======================" << endl;
    if(horizontal) {
        for(uint32_t i = 0; i < size; i++) {
            cout << i << ". " << x[i] << "\t";
        }
        cout << endl;
    }
    else {
        for(uint32_t i = 0; i < size; i++) {
            cout << i << ". " << x[i] << endl;
        }
    }
    cout << "==============================================================" << endl;
}

void Print1dNumpyFriendlyArray(string const &str1, double* x, uint32_t size) {
    cout << "======================= " << str1 << " =======================" << endl;
    cout << "[";
    for(uint32_t i = 0; i < size - 1; i++) {
        cout << x[i] << ",";
    }
    cout << x[size - 1] << "]" << endl;
    cout << "==============================================================" << endl;
}

void PrintValue(string const &str1, double x) {
    cout << "======================= " << str1 << " =======================" << endl;
    cout << x << endl;
    cout << "==============================================================" << endl;
}


void Print1dMatrixByWindows(string const &str1, double *matrix, uint32_t m_row, uint32_t m_col, uint32_t w_row,
                            uint32_t w_col) {
    cout << "======================= " << str1 << " =======================" << endl << endl;
    for(uint32_t i = 0; i < m_row; i++) {
        //delimiter between windows in horizontal direction
        if(i % w_row == 0){
            for(uint32_t d = 0; d < m_col; d++){
                cout << " _ _ _ _ _ _ _ _";
            }
            cout << endl;
        }
        for(uint32_t j = 0; j < m_col; j++){
            //delimiter between windows in vertical direction
            if(j % w_col == 0){
                cout << "|\t";
            }
            cout << matrix[i*m_col + j] << " \t " ;
        }
        cout << "|" << endl;
    }
    for(uint32_t d = 0; d < m_col; d++){
        cout << " _ _ _ _ _ _ _ _";
    }
    cout << endl << "==============================================================" << endl;
}

// Matrix operations
static double** MultiplyMatrices(double** m1, double** m2, int m1_row, int m1_col, int m2_col) {
    double tmp;
    double** res = new double*[m1_row];
    for( int i = 0; i < m1_row; i++) {
        res[i] = new double[m2_col];
        for(int j = 0; j < m2_col; j++) {
            tmp = 0;
            for(int k = 0; k < m1_col; k++) {
                tmp += m1[i][k] * m2[k][j];
            }
            res[i][j] = tmp;
        }
    }
    return res;
}

static double* MultiplyMatrixVector(double** m1, double* m2, int m1_row, int m1_col) {
    double tmp;
    double* res = new double[m1_row];
    for( int i = 0; i < m1_row; i++) {
        tmp = 0;
        for(int k = 0; k < m1_col; k++) {
            tmp += m1[i][k] * m2[k];
        }
        res[i] = tmp;
    }
    return res;
}

double*** DoubleMatrixMatrixMultiply(double ***a, double ***b, uint32_t n_mats, uint32_t a_row, uint32_t a_col, uint32_t b_col) {
    /*
     * Perform several multiplication of double matrices a and b. The function assumes that the number of columns of a equals to
     * the number of rows of b.
     *
     * Input(s)
     * a: three dimensional matrix of size n_mats-by-a_row-by-a_col
     * b: three dimensional matrix of size n_mats-by-a_col-by-b_col
     *
     * Output(s)
     * Returns a matrix of size n_mats-by-a_row-by-b_col
     */
    if(DEBUG_FLAG == 1)
        cout << "************************************************************\nDoubleMatrixMatrixMultiply is called" << endl;
    double ***result = new double **[n_mats];
    for(uint32_t g = 0; g < n_mats; g++) {
        result[g] = new double*[a_row];
        double tmp_sum = 0;
        for (uint32_t i = 0; i < a_row; i++) {
            result[g][i] = new double[b_col];
            for (uint32_t j = 0; j < b_col; j++) {
                tmp_sum = 0;
                for (uint32_t k = 0; k < a_col; k++) {
                    tmp_sum += a[g][i][k] * b[g][k][j];
                }
                result[g][i][j] = tmp_sum;
            }
        }
    }
    if(DEBUG_FLAG == 1)
        cout << "Returning from DoubleMatrixMatrixMultiply...\n************************************************************" << endl;
    return result;
}

static void BubbleSort(double x[], int n) {
    bool exchanges;
    do {
        exchanges = false;  // assume no exchanges
        for (int i=0; i<n-1; i++) {
            if (x[i] > x[i+1]) {
                double temp = x[i]; x[i] = x[i+1]; x[i+1] = temp;
                exchanges = true;  // after exchange, must look again
            }
        }
    } while (exchanges);
}

/* Functions to read data with various sizes
 * They read arrays with different dimensionalities.
 */
double** Read2dArrayGt(const string& fn, int n_anc, int n_dim, int k_mer) {
    /*  This function reads the anchor points.
     *  Input(s):
     *  fn: file name
     *  n_anc: number of the anchor points
     *  n_dim: number of dimension to represent each character in the anchor points
     *  k_mer: number of layers or utilized lenght of k-mers
     *
     *  Return(s):
     *  the anchor points in two dimensional double dynamic array of size n_anc-by-n_dim
     */
    // File pointer
    fstream fin;

    // Open an existing file
    fin.open(fn, ios::in);

    double** anchor_points = new double*[n_anc];
    string word, temp;
    int anc_ind = 0, dim_ind = 0;
    while (fin >> temp) {
        // used for breaking words
        stringstream s(temp);

        // read every column data of a row and store it in a string variable, 'word'
        anchor_points[anc_ind] = new double[n_dim];
        while (getline(s, word, ',')) {
            anchor_points[anc_ind][dim_ind] = stof(word);
            dim_ind++;
        }

        anc_ind++;
        dim_ind = 0;
    }

    return anchor_points;
}

double* Read1dArrayGt(const string& fn, int n_anc, bool bias_flag = true) {
    /*  This function reads the weights in the linear classifier layer as well as the bias in this layer.
     *  Input(s):
     *  fn: file name
     *  n_anc: number of the anchor points
     *  bias_flag: whether there is bias term at the end of the given file
     *
     *  Return(s):
     *  the weights in one dimensional double dynamic array of size n_anc or (n_anc + 1) depending on bias_flag
     */
    cout << "File name: " << fn << endl;
    // File pointer
    fstream fin;

    // Open an existing file
    fin.open(fn, ios::in);

    double* weights;
    if(bias_flag) {
        weights = new double[n_anc + 1];
    }
    else {
        weights = new double[n_anc];
    }

    string word, temp;
    int ind = 0;

    // read the single line
    getline(fin, temp);

    // used for breaking words
    stringstream s(temp);

    // read every column data of a row and
    // store it in a string variable, 'word'
    while (getline(s, word, ',')) {
        weights[ind] = stof(word);
        ind++;
    }

    return weights;
}

uint64_t** Read2dArray(Party* proxy, const string& fn, int n_anc, int n_dim) {
    /*  This function reads the anchor points.
     *  Input(s):
     *  proxy: Party instance
     *  fn: file name
     *  n_anc: number of the anchor points
     *  n_dim: number of dimension to represent each character in the anchor points
     *
     *  Return(s):
     *  shares of the anchor points in two dimensional uint64_t dynamic array of size n_anc-by-n_dim
     */
    // File pointer
    fstream fin;

    // Open an existing file
    cout << "File name: " << fn << endl;
    fin.open(fn, ios::in);
    if (fin.is_open()) {
        cout << "File successfully open" << endl;
    }
    else {
        cout << "Error opening file" << endl;
        return nullptr;
    }

    uint64_t** anchor_points = new uint64_t*[n_anc];
    string word, temp;
    int anc_ind = 0, dim_ind = 0;
    int cnt = 0;

    while (fin >> temp) {
        cnt++;
        // used for breaking words
        stringstream s(temp);

        // read every column data of a row and store it in a string variable, 'word'
        anchor_points[anc_ind] = new uint64_t[n_dim];
        while (getline(s, word, ',')) {
            uint64_t tmp;
            if(proxy->GetPRole() == proxy1) {
                tmp  = ConvertToUint64(stof(word)) - proxy->GenerateCommonRandom();
            }
            else {
                tmp = proxy->GenerateCommonRandom();
            }
            anchor_points[anc_ind][dim_ind] = tmp;
            dim_ind++;
        }

        anc_ind++;
        dim_ind = 0;
    }

    return anchor_points;
}

uint64_t* Read1dArray(Party* proxy, const string& fn, int n_anc, bool bias_flag = true) {
    /*  This function reads the weights in the linear classifier layer as well as the bias in this layer.
     *  Input(s):
     *  proxy: Party instance
     *  fn: file name
     *  n_anc: number of the anchor points
     *  bias_flag: whether there is bias term at the end of the given file
     *
     *  Return(s):
     *  shares of the weights in one dimensional uint64_t dynamic array of size n_anc or (n_anc + 1) depending on bias_flag
     */
    cout << "File name: " << fn << endl;
    // File pointer
    fstream fin;

    // Open an existing file
    fin.open(fn, ios::in);

    uint64_t* weights;
    if(bias_flag) {
        weights = new uint64_t[n_anc + 1];
    }
    else {
        weights = new uint64_t[n_anc];
    }

    string word, temp;
    int ind = 0;

    // read the single line
    getline(fin, temp);

    // used for breaking words
    stringstream s(temp);

    // read every column data of a row and
    // store it in a string variable, 'word'
    while (getline(s, word, ',')) {
//        cout << "Index: " << ind << "\tWord: " << word << endl;
        if(proxy->GetPRole() == proxy1) {
            weights[ind] = ConvertToUint64(stof(word)) - proxy->GenerateCommonRandom();
        }
        else {
            weights[ind] = proxy->GenerateCommonRandom();
        }
        ind++;
    }

    return weights;
}

/*
 * Specific matrix/vector generation functions
 */
static uint64_t* Zero1dData(Party *proxy, int size) {
    uint64_t* mat_data = new uint64_t[size];
    for (int i = 0; i < size; i++) {
        // generate shares
        uint64_t val = 0;
        for (int k = 3; k >= 0; k -= 1) {
            uint64_t a = rand() & 0xffff;
            val = val ^ (a << (k * 16));
        }

        if (proxy->GetPRole() == 0) {
            mat_data[i] = val;
        } else {
            mat_data[i] = 0 - val;
        }
    }
    return mat_data;
}

//******************************************************************************
// Rest

string valid_alphabet = "ARNDCQEGHILKMFPSTWYV";
string invalid_alphabet = "XBZJUO";

static void VectorisedBubbleSort(double **x, int r, int c) {
    for(int i = 0; i < r; i++) {
        BubbleSort(x[i], c);
    }
}

static double** InplaceDotProduct(double** m1, double** m2, int n_row, int n_col) {
    double tmp;
    double** res = new double*[n_row];
    for( int i = 0; i < n_row; i++) {
        res[i] = new double[n_row];
        for(int j = 0; j < n_row; j++) {
            tmp = 0;
            for(int k = 0; k < n_col; k++) {
                tmp += m1[i][k] * m2[j][k];
            }
            res[i][j] = tmp;
        }
    }
    return res;
}

static double MultiplyVectorVector(double* m1, double* m2, int length) {
    double tmp = 0;
    for( int i = 0; i < length; i++) {
        tmp += m1[i] * m2[i];
    }
    return tmp;
}

string RecoverSequence(const string& fn, int index) {
    /*
     * Recover a sequence based on the indices read from the specified file
     */
    // File pointer
    fstream fin_pre;
    fin_pre.open(fn, ios::in);
    if (fin_pre.is_open()) {
        cout << "File successfully open in RecoverSequence" << endl;
    }
    else {
        cout << "Error opening file in RecoverSequence" << endl;
        return "ERROR";
    }
    string word, temp;
    for(int i = 0; i <= index; i++) {
        getline(fin_pre, temp);
    }
    stringstream s(temp);

    // to identify the last nonzero entry and meanwhile store the values
    char delimiter = ',';
    int ind = 0;
    int last_nonzeros = 0;
    char seq_pre[10000];
    while (getline(s, word, delimiter)) { // && ind < size
        int tmp = stoi(word) - 1;
        if(tmp < 0) {
            last_nonzeros = ind;
            break;
        }
        seq_pre[ind] = valid_alphabet[tmp];
        ind++;
    }
    cout << endl;

    // just extract the characters until the last nonzero character
    char* seq = new char[last_nonzeros]();
    for(int i = 0; i < last_nonzeros; i++) {
        seq[i] = seq_pre[i];
    }

    string str(seq, last_nonzeros);

    return str;
}

unordered_map<char, double*> GenerateOneHotMapping() {
    /*  This function generates a mapping for the characters that one can encounter in a protein sequence.
     *
     *  Input(s):
     *
     *  Returns(s):
     *  An unordered_map whose keys are characters and the values are double dynamic array of size 20
     */
    unordered_map<char, double*> one_hot_mapping;

    // valid characters
    for(int i = 0; i < valid_alphabet.length(); i++) {
        double* at = new double[valid_alphabet.length()];
        for(int j = 0; j < valid_alphabet.length(); j++) {
            if(i == j) {
                at[j] = 1;
            }
            else {
                at[j] = 0;
            }
        }
        one_hot_mapping[valid_alphabet[i]] = at;
    }

    // invalid characters
    for(char & i : invalid_alphabet) {
        double* at = new double[valid_alphabet.length()];
        for(int j = 0; j < valid_alphabet.length(); j++) {
            at[j] = 1.0 / valid_alphabet.length(); // the mapping of the invalid characters is 1 / size_of_valid_alphabet
        }
        one_hot_mapping[i] = at;
    }

    return one_hot_mapping;
}

uint64_t** EncodeSequence(Party* proxy, string seq) {
    /*  This function encodes the sequence via one-hot encoding and returns the secret shared form of the two dimensional
     *  uint64_t array.
     *
     *  Input(s):
     *  proxy: Party instance to generate common random values
     *  seq: sequence as a string
     *
     *  Return(s):
     *  Secret shared form of one-hot encoding of the given sequence as the two dimensional uint64_t array
     */

    unordered_map<char, double*> one_hot_mapping = GenerateOneHotMapping();

    uint64_t** x_share = new uint64_t*[seq.length()];
//    cout << "seq_processing::EncodeSequence::sequence length: " << seq.length() << endl;
    for(int i = 0; i < seq.length(); i++) {
        // check if the character in the sequence exists in the mapping
        if(one_hot_mapping.find(seq[i]) == one_hot_mapping.end()) {
            cout << "Sequence has an invalid character \"" << seq[i] << "\" at index " << i << endl;
            throw;
        }

        double* tmp = one_hot_mapping[seq[i]];
        x_share[i] = proxy->CreateShare(tmp, 20);
    }

    return x_share;
}

#endif //CECILIA_TEST_FUNCTIONS_H
