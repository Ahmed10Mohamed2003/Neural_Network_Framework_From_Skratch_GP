#include "trainSet.h"

trainSet::trainSet()
{
}
////////////////////////////////////////////////////////////////////////
void trainSet::Create()
{   /*x    = matD(nPat,nIn);       // full input data
    y    = matD(nPat,nOut);      // full output data*/
    MinistConnection();
}
void trainSet::LoadFromBinary(const char* filename)
{
    ifstream in(filename, ios::binary);
    if (!in)
    {
        cout << "Error: cannot open training set file\n";
        exit(1);
    }

    // 1. Read sizes
    in.read((char*)&nPat, sizeof(int));
    in.read((char*)&nIn,  sizeof(int));
    in.read((char*)&nOut, sizeof(int));

    // 2. Allocate memory
    Create();

    // 3. Read X
    for (int p = 0; p < nPat; p++)
        in.read((char*)x[p], sizeof(double) * nIn);

    // 4. Read Y
    for (int p = 0; p < nPat; p++)
        in.read((char*)y[p], sizeof(double) * nOut);

    in.close();

    cout << "Training set loaded:\n";
    cout << "Patterns = " << nPat
         << ", Inputs = " << nIn
         << ", Outputs = " << nOut << endl;
}

////////////////////////////////////////////////////////////////////////
void trainSet::XfillRand( int p) // to fill x with random variable
{   for(int j=0; j<nIn; j++)
        for(int n=0; n<nPat; n++)
        {   double  r = rand() % 100;
            if(r>p) r = 1;
            else    r = -1;
            x[n][j]   = r;
        }
}
////////////////////////////////////////////////////////////////////////
void trainSet::YfillParity () // to fill y with parity.
{   int i,j,pluss;
    for(j=0; j<nPat; j++)
    {   pluss=0;
        for(i=0; i<nIn; i++)
            if(x[j][i]==1)   pluss+=1;
        y[j][0]=-1;
        for(i=1; i<=nIn; i+=2)
            if  (pluss==i)  y[j][0]=1;
    }
}
////////////////////////////////////////////////////////////////
void  trainSet::printTs()
{   int i,j;
    cout<<"Ts="<<endl;
    for(i=0; i<nPat; i++)
    {   for(j=0; j<nIn; j++)
            cout<<setw(3)<<x[i][j]<<" ";
        cout<<"="<<setw(9);
        for(j=0; j<nOut; j++)   cout<<setw(3)<<y[i][j];
        cout<<endl;
    }
}

/////////////////////My_ADDs//////////////////////////////////

void trainSet::XfillFromFile(const char* filename)
{
    ifstream fin(filename);
    if (!fin) {
        cout << "Error opening file!\n";
        return;
    }

    int file_nPat, file_nIn, file_nOut;
    fin >> file_nPat >> file_nIn >> file_nOut;

    if (file_nPat != nPat || file_nIn != nIn) {
        cout << "Dimension mismatch!\n";
        fin.close();
        return;
    }

    // Read X directly (>> automatically skips comments & whitespace)
    for (int i = 0; i < nPat; i++)
        for (int j = 0; j < nIn; j++)
            fin >> x[i][j];

    fin.close();
}

void trainSet::YfillFromFile(const char* filename)
{
    ifstream fin(filename);

    if (!fin.is_open()) {
        cout << "Error opening file!\n";
        return;
    }

    int file_nPat, file_nIn, file_nOut;
    fin >> file_nPat >> file_nIn >> file_nOut;

    if (file_nPat != nPat) {
        cout << "Pattern mismatch in Y file!\n";
        fin.close();
        return;
    }

    // skip X rows
    double temp;
    for (int i = 0; i < nPat; i++)
        for (int j = 0; j < file_nIn; j++)
            fin >> temp;

    // read Y
    for (int i = 0; i < nPat; i++)
        for (int o = 0; o < file_nOut; o++)
            fin >> y[i][o];

    fin.close();
}
unsigned int readI(ifstream* inDataFile)
{   unsigned char a,b,c,d;
    unsigned int r=0;
    inDataFile->read((char*)(&a), sizeof(char));
    inDataFile->read((char*)(&b), sizeof(char));
    inDataFile->read((char*)(&c), sizeof(char));
    inDataFile->read((char*)(&d), sizeof(char));
    r=d+256*c+65536*b+16777216*a;
    return r;
}
/////////////////////////////readIm/////////////////////////////////////
double ** readIm(unsigned int& r, unsigned int& w,unsigned int& patternLen,
                 unsigned int& nPat, char* name)
{   unsigned int mn;
    unsigned char* p;

    ifstream imF;
    imF.open(name,ios::binary|ios::in);
    if (!imF)
    {   cout << "Unable to open file TRIMG";
        exit(1);   // call system to stop
    }
    mn=readI(&imF);
    nPat=readI(&imF);
    r=readI(&imF);
    w=readI(&imF);
    patternLen=r*w;
    p =(unsigned char *) malloc(patternLen*nPat);
    imF.read((char*)(p), patternLen*nPat);
    imF.close();
    cout<<mn<<"  "<<nPat<<"  "<<r<<"  "<<w<<endl;
    //====================all data flat but double======================
    double *  pat= (double *) malloc(sizeof(double)*patternLen*nPat);
    unsigned int i;
    for (i=0; i<patternLen*nPat; i++) pat[i]=(p[i]-127.5)/127.5; // from -1 to 1
    //==================== as a two dimensional array====================
    double ** patMat=(double**) malloc(sizeof(double*)*nPat);
    for (i=0; i<nPat; i++) patMat[i]= pat+i*patternLen;
    delete [] p;
    //never do "delete [] pat;"   it is the only place with pattern data
    return patMat;
}
double ** readLABEL(unsigned int& r, unsigned int& w, char* name)
{   unsigned int mn,nPat=0;;
    unsigned char* imL;

    ifstream labF;
    labF.open(name,ios::binary|ios::in);
    if (!labF)
    {   cout << "Unable to open file LABEL";
        exit(1);   // call system to stop
    }
    mn=readI(&labF);
    nPat=readI(&labF);
    imL= (unsigned char*) malloc(nPat);
    labF.read((char*)(imL), sizeof(char)*nPat);
    labF.close();
    //====================all data flat but double======================
    double *  label= (double *) malloc(sizeof(double)*10*nPat);
    cout<<mn<<"  "<<nPat<<"  "<<r<<"  "<<w<<endl;
    unsigned int i, j;
    for (i=0; i<nPat; i++)
        for (j=0; j<10; j++)
        {   if (j==imL[i]) label[10*i+j]=1.0;
            else label[10*i+j]=-1.0;
        }
    //==================== as a two dimensional array====================
    double **labMat=(double**) malloc(sizeof(double*)*nPat);
    for (i=0; i<nPat; i++) labMat[i]= label+i*10;
    delete []imL;
    //never do "delete [] label;"   it is the only place with pattern data

    return labMat;
}
void trainSet::MinistConnection()
{
    unsigned int k,r,w;
    unsigned int patternLen =0, nPat=0;
    char IMFname[]="TRIMG";
    char LFname []="LABEL";
    x =  readIm   (r,w,patternLen,nPat,IMFname);
    cout<< "<><><><><><><><><><><><>><><><>><><>"<<endl;
    y =  readLABEL(r,w, LFname);
    k=0;
    cout<< "<><><><><><><><><><><><>><><><>><><>"<<endl;
   /* while (k<nPat)
    {   displayIm(P[k],L[k]);
        k++;
        //getche();
    }*/
    //savePL("Test.bin", P, L, nPat, patternLen, 10);
}

// Helper function to shift a single image
double* trainSet::shiftImageOnce(double* image, int rows, int cols, int shiftX, int shiftY)
{
    // Allocate new image
    double* shifted = new double[rows * cols];

    // Initialize with background value (-1 for MNIST)
    for (int i = 0; i < rows * cols; i++)
        shifted[i] = -1.0;

    // Copy pixels with shift
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int newR = r + shiftY;
            int newC = c + shiftX;

            // Check bounds
            if (newR >= 0 && newR < rows && newC >= 0 && newC < cols)
            {
                shifted[newR * cols + newC] = image[r * cols + c];
            }
        }
    }

    return shifted;
}

// Augment dataset - adds ONE shifted version per image
void trainSet::augmentDataByShifting(int maxShift)
{
    cout << "Augmenting dataset with single random shift per image..." << endl;
    cout << "Original patterns: " << nPat << endl;

    int originalPat = nPat;
    int rows = 28;
    int cols = 28;

    // Save old pointers
    double** oldX = x;
    double** oldY = y;

    // Double the dataset
    int augmentedPat = originalPat * 2;

    // Create new arrays
    double** newX = new double*[augmentedPat];
    double** newY = new double*[augmentedPat];

    srand(time(0));

    int idx = 0;

    for (int p = 0; p < originalPat; p++)
    {
        // 1. Copy original image (just copy the pointer, don't allocate new)
        newX[idx] = oldX[p];  // Reuse original data
        newY[idx] = oldY[p];  // Reuse original labels
        idx++;

        // 2. Add ONE shifted version (allocate new memory for this)
        int shiftX = (rand() % (2 * maxShift + 1)) - maxShift;
        int shiftY = (rand() % (2 * maxShift + 1)) - maxShift;

        newX[idx] = shiftImageOnce(oldX[p], rows, cols, shiftX, shiftY);

        // Copy label
        newY[idx] = new double[nOut];
        for (int o = 0; o < nOut; o++)
            newY[idx][o] = oldY[p][o];
        idx++;

        // Progress
        if (p % 10000 == 0)
            cout << "Processed " << p << " / " << originalPat << endl;
    }

    cout << "Augmentation complete!" << endl;

    // DON'T free oldX/oldY - they were allocated with malloc
    // Just update the pointers
    x = newX;
    y = newY;
    nPat = augmentedPat;

    cout << "New pattern count: " << nPat << " (2x original)" << endl;
}
