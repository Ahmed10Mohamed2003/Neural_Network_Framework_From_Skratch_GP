#include "layer.h"

layer:: layer(int myin, int myout, double myalfa,int* myPnPatern)
    :nIn(myin), nOut(myout),alfa(myalfa), pnPat(myPnPatern)
{   int i, j;
    w     = matD(nOut,nIn);
    dw    = matD(nOut,nIn);
    b     = matD(nOut);
    db    = matD(nOut);
    vw = matD(nOut, nIn);
    vb = matD(nOut);
    beta=0.9;

    mOutF = matD(nOut);
    mOutB = matD(nIn);

	for(i=0; i<nIn; i++)   for(j=0; j<nOut; j++) dw[j][i]=0;
    for(j=0; j<nOut; j++)   db[j]=0;

    for(i=0; i<nOut; i++)
    {   for(j=0; j<nIn; j++)
            w[i][j]=((rand()%100)-50)/50.00;
        b[i]=((rand()%100)-50)/50.00;
    }

    for (int j = 0; j < nOut; j++)
    {
    vb[j] = 0;
    for (int i = 0; i < nIn; i++)
        vw[j][i] = 0;
    }
}
////////////////////////////////////////////////////////////////////
void layer::BP ()
{   int i,j,k;
    double dz;
//-------------------initialization----------------------
   for(i=0; i<nIn; i++)mOutB[i]=0;

//-----------------------main loops----------------------
   for(i=0; i<nOut; i++)
   {
	   dz = pInB[i] * (1.1 - (mOutF[i]*mOutF[i]));
	   db[i]+= dz;
	   for(j=0; j<nIn; j++)
	   {
		   dw[i][j]+= dz * pInF[j];
		   mOutB[j]+=dz * w[i][j];
	   }
   }
//--------------------------update----------------------------

}
void layer::update()
{
	for(int j=0; j<nOut; j++)
    {
       db[j]/=64;
       vb[j] = beta * vb[j] + db[j];
        b[j]  += alfa * vb[j];
        for(int i=0; i<nIn; i++)
{
    double decay = 0.0002;

    // 1. Average gradient over batch
    dw[j][i] /= 64;

    // 2. Update momentum (velocity)
    vw[j][i] = beta * vw[j][i] + dw[j][i];

    // 3. Apply weight decay + momentum update together
    w[j][i] = w[j][i] * (1 - alfa * decay) + alfa * vw[j][i];
}
    }
	for(int i=0; i<nIn; i++)   for(int j=0; j<nOut; j++) dw[j][i]=0;
    for(int j=0; j<nOut; j++)   db[j]=0;
}
////////////////////////////////////////////////////////////////////
void layer::FF() // to get layer output
{   int i,j,k;
    double z,a;

    for(i=0; i<nOut; i++)
	{
		z=0;
		for(j=0; j<nIn; j++)
		{
			z+= pInF[j] * w[i][j];
		}
		z+=b[i];
		//should be  mOutF[k][j]=tanh(z);
        // but the following is better
        if (z>1)a=1;
        else if(z<-1)a=-1;
        else a=z;
		mOutF[i]=a;
	}
}
//////////////////////////////////////////////////////////////////////
void layer::makeBefore(layer* L) //connect phantom before L
{   pInB=L->mOutB;
    L->pInF= mOutF;
}
//////////////////////////////////////////////////////////////////////
void layer::print()
{   int i,j;
    //int nPat=*tr->pnPat;
    cout << "No of Input    ="<< nIn  <<  endl;
    cout << "No of Output   ="<< nOut <<  endl;
    cout << "Alfa Value     ="<< alfa <<  endl;

    for(i=0; i<nOut  ; i++)
    {   cout<<"w["<<i+1<<"] = ";
        for(j=0; j<nIn; j++) cout<<w[i][j]<<" , ";
        cout<< "b["<<i+1<<"] = "<<b[i]<<endl;
    }
}
