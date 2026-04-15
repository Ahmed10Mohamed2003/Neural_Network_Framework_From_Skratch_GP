#ifndef TRAINER_H
#define TRAINER_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <conio.h>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "layer.h"
#include "trainSet.h"
#include "net.h"

using namespace std;
class trainSet;
class layer;
class net;

const int Cont=0;  // normal
const int Done=1;  // we are done

class trainer
{
public:
    trainSet* ts;       // pointer to training set.
    net* Net;           // the net
    double MaxError;    // maximum error
    double Loss;        // sum of all error
    int errorCount;     // number of pattern in error
    double * pa;       // pointer to the output of the last layer
    double * mda;      // de/do
    int mode;           // cont or done

    // pointer for other variables to improve computation
    int* pnIn;           // pointer to number of input,
    int* pnOut;          // pointer to number of output,
    int* pnPat;          // pointer to number of patterns
    double ** px;        // pointer to input
    double ** py;        // pointer to output

    trainer(net* theNet,trainSet* ts);
    void   NFF ();       // network feed forward
    void   NBP ();       // network back propagation
    void   update(int j);     // to update de/do and others
    void   printTs_out();
    int    train (int cycles);
    void   Test();
	void   saveWeights();
	void   loadWeights();
	void   save_vw_vb();
	void   load_vw_vb();
	int    predictDigit(double* out, int nOut);
	int    trueDigit(double* label, int nOut);
};

#endif // TRAINER_H

