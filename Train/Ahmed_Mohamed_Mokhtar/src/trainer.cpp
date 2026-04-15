#include "trainer.h"

trainer::trainer(net* theNet,trainSet* myts): ts(myts), Net(theNet)
{   pnIn  = &(ts->nIn);
    pnOut = &(ts->nOut);
    pnPat = &(ts->nPat);
    mda   =  matD(*pnOut);   // de/do
    px = (ts->x);
    py = (ts->y);
	//errorCount= 0;
    //Net->Ls[0]->pInF = px;
    // to connect trainer to the network and training set.
    Net->Ls[Net->nL-1]->pInB  = mda;      // connect input for last layer for BP
    pa = Net->Ls[Net->nL-1]->mOutF;
}
//////////////////////////////////////////////////////////////////////////////////////

/*void trainer::update(int j)
{   int i,k;
    int nError=0;      // number of outputs with error
    double error=0;
    int riqured_indx = 0;
 int O_index = 0;
 int large_out = pa[0];
 for(i=0; i<*pnOut; i++)
 {
  mda[i] = (py[j][i]-pa[i]);
  if(pa[i] > large_out && pa[i]>0)
  {
   large_out = pa[i];
   O_index = i;
  }
  if(py[j][i] == 1) riqured_indx = i;

 }

 if (riqured_indx != O_index)    //  pattern still has error
    {   errorCount++;
    }
}*/

void trainer::update(int j)
{
    int i;
    int O_index = 0;
    int riqured_indx = 0;
    double large_out = -999999.0; // Use a very small double

    for(i = 0; i < *pnOut; i++)
    {
        // 1. Calculate the delta for backprop
        mda[i] = (py[j][i] - pa[i]);

        // 2. Find the index of the highest activation (Prediction)
        if(pa[i] > large_out)
        {
            large_out = pa[i];
            O_index = i;
        }

        // 3. Find the index of the target (Ground Truth)
        if(py[j][i] == 1)
        {
            riqured_indx = i;
        }
    }

    // 4. Update error count if prediction is wrong
    if (riqured_indx != O_index)
    {
        errorCount++;
    }
}
//////////////////////////////////////////////////////////////////////////////////////
/*void trainer::update(int j)
{  int i,k;

    double target=py[j][0];
    double target_o=pa[0];
    int index=0;
    int index_o=0;
    for(i=0; i<*pnOut; i++)
    {
        if(py[j][i]>target)
            {
                index=i;
                target=py[j][i];
            }
        if(pa[i]>target_o)

            {
                index_o=i;
                target_o=pa[i];
            }
    }

	if (index!=index_o)    //  pattern still has error
    {   errorCount++;
    }
*/
/*void trainer::update(int j)
{  int i,k;
    int nError=0;      // number of outputs with error
    double error=0;

	for(i=0; i<*pnOut; i++)
	{
		mda[i] = (py[j][i]-pa[i]);
		error = abs(mda[i]);
		Loss += error;
		if (error>MaxError) MaxError=error;
            nError += error>0.3;
	}
	if (nError>0)    //  pattern still has error
    {   errorCount++;
    }
}*/
////////////////////////////////////////////////////////////////
void  trainer::printTs_out()
{   /*int i,j;
    cout<<"Ts="<<endl;
    for(i=0; i<*pnPat; i++)
    {   for(j=0; j<*pnIn; j++)
            cout<<setw(3)<<px[i][j]<<" ";
        cout<<"=";
        for(j=0; j<*pnOut; j++)   cout<<setw(3)<<py[i][j];
        for(j=0; j<*pnOut; j++)   cout<<setprecision (3)<<setw(5)<<pa[i][j]<<" E=";
        for(j=0; j<*pnOut; j++)   cout<<setprecision (3)<<setw(5)<<abs(pa[i][j]-py[i][j]);
        cout<<endl;
    }*/
}
////////////////////////////////////////////////////////////////
void  trainer:: NFF ()
{   int i;
    // Net->nL : number of layers in the net
    for (i=0; i<Net->nL; i++ ) (*Net)[i]->FF();
}
////////////////////////////////////////////////////////////////
void  trainer:: NBP ()
{   int i;
    // Net->nL : number of layers in the net
    for (i=Net->nL-1; i>=0; i-- )  (*Net)[i]->BP();
}
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
int	trainer::train (int cycles)
{
    int i,j,k,p,batchSize=64;

	int* indices = new int[*pnPat];

	for (i = 0; i < *pnPat; i++) indices[i] = i;

	for(i=0; i<cycles; i++)
	{

	    if (i%120==0)
        {
            int ch;
            cout<<"\ncontinue 1 stop else\n";
            cin>>ch;
            if(ch!=1)
                {
                    saveWeights();
                    save_vw_vb();
                    exit(1);

                }

        }

		MaxError=0;
		Loss=0;
		errorCount= 0;

		// 1. Shuffle patterns
		for ( k = *pnPat - 1; k > 0; k--)
		{
			p = rand() % (k + 1);
			swap(indices[k], indices[p]);
		}
		for (int batchStart = 0; batchStart < *pnPat; batchStart += batchSize)
		{
			int batchEnd = min(batchStart + batchSize, *pnPat);

			// Loop patterns in batch
			for ( k = batchStart; k < batchEnd; k++)
			{
				int idx = indices[k];
				Net->Ls[0]->pInF = px[idx];
				NFF();
				update(idx);
				NBP();
			}

			// Update weights after batch
			for (int L = 0; L < Net->nL; L++)
				Net->Ls[L]->update();
		}

		Loss/=(*pnPat);
		 cout<<"i="<<i<<"  errorCount= "<<errorCount<<endl;
		if(errorCount== 0)
		{   mode= Done;     // we are done
			break;
		}
	}
	save_vw_vb();
	saveWeights();
	delete[] indices;
	return i;
}
//////////////////////////////////////////////////////////////////////////////////////
void trainer:: Test(void)
{
    errorCount=0;
    for (int i = 0; i < *pnPat; i++)
    {
        Net->Ls[0]->pInF = px[i];
        NFF();
        update(i);
    }
    cout<<" errorCount = "<<errorCount<<endl;
}
//////////////////////////////////////////////////////////////////////////////////////
void trainer::saveWeights()
{
    int i, j, k;
    // Remove ios::binary for text mode
    ofstream file("weights.txt", ios::trunc);

    if (!file.is_open())
    {
        cout << "Error: Could not open file weights.txt" << endl;
        return;
    }

    file << Net->nL << endl;  // number of layers

    for(i = 0; i < Net->nL; i++)
    {
        file << i << endl;
        layer* temp = (*Net)[i];

        file << temp->nOut << " " << temp->nIn << endl;

        for(j = 0; j < temp->nOut; j++)
        {
            // Write all weights for this neuron
            for(k = 0; k < temp->nIn; k++)
            {
                file << temp->w[j][k] << " ";
            }
            // Write bias with space after it
            file << temp->b[j] << " ";
            file << endl;  // Newline after each neuron
        }
    }

    file.close();
    cout << "Weights saved to weights.txt" << endl;
}
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void trainer::loadWeights()
{
    int i, j, k;

    ifstream file("weights.txt");

    if (!file.is_open())
    {
        cout << "Error: Could not open file weights.txt for reading" << endl;
        return;
    }

    int nLayers;
    file >> nLayers;

    if (nLayers != Net->nL)
    {
        cout << "Error: Layer count mismatch. File has " << nLayers
             << " layers, network has " << Net->nL << endl;
        file.close();
        return;
    }

    for(i = 0; i < Net->nL; i++)
    {
        int layerIndex;
        file >> layerIndex;

        layer* temp = (*Net)[i];

        int nOut, nIn;
        file >> nOut >> nIn;

        if (nOut != temp->nOut || nIn != temp->nIn)
        {
            cout << "Error: Layer " << i << " dimension mismatch!" << endl;
            cout << "Expected: " << temp->nOut << " x " << temp->nIn << endl;
            cout << "Found in file: " << nOut << " x " << nIn << endl;
            file.close();
            return;
        }

        // Read weights and biases
        for(j = 0; j < temp->nOut; j++)
        {
            for(k = 0; k < temp->nIn; k++)
            {
                if (!(file >> temp->w[j][k]))
                {
                    cout << "Error reading weight[" << j << "][" << k << "]" << endl;
                    file.close();
                    return;
                }
            }

            if (!(file >> temp->b[j]))
            {
                cout << "Error reading bias[" << j << "]" << endl;
                file.close();
                return;
            }
        }
    }

    file.close();
    cout << "Weights loaded successfully from weights.txt" << endl;
}
void   trainer::save_vw_vb()
{
 int i, j, k;
    // Remove ios::binary for text mode
    ofstream file("vwvb.txt", ios::trunc);

    if (!file.is_open())
    {
        cout << "Error: Could not open file vwvb.txt" << endl;
        return;
    }
 file << Net->nL << endl;  // number of layers

    for(i = 0; i < Net->nL; i++)
    {
        file << i << endl;
        layer* temp = (*Net)[i];

        file << temp->nOut << " " << temp->nIn << endl;

        for(j = 0; j < temp->nOut; j++)
        {
            // Write all weights for this neuron
            for(k = 0; k < temp->nIn; k++)
            {
                file << temp->vw[j][k] << " ";
            }
            // Write bias with space after it
            file << temp->vb[j] << " ";
            file << endl;  // Newline after each neuron
        }
    }

    file.close();
    cout << "vwvb saved to vwvb.txt" << endl;

}
void   trainer::load_vw_vb()
{
 int i, j, k;

    ifstream file("vwvb.txt");

    if (!file.is_open())
    {
        cout << "Error: Could not open file vwvb.txt for reading" << endl;
        return;
    }

    int nLayers;
    file >> nLayers;

    if (nLayers != Net->nL)
    {
        cout << "Error: Layer count mismatch. File has " << nLayers
             << " layers, network has " << Net->nL << endl;
        file.close();
        return;
    }

    for(i = 0; i < Net->nL; i++)
    {
        int layerIndex;
        file >> layerIndex;

        layer* temp = (*Net)[i];

        int nOut, nIn;
        file >> nOut >> nIn;

        if (nOut != temp->nOut || nIn != temp->nIn)
        {
            cout << "Error: Layer " << i << " dimension mismatch!" << endl;
            cout << "Expected: " << temp->nOut << " x " << temp->nIn << endl;
            cout << "Found in file: " << nOut << " x " << nIn << endl;
            file.close();
            return;
        }

        // Read weights and biases
        for(j = 0; j < temp->nOut; j++)
        {
            for(k = 0; k < temp->nIn; k++)
            {
                if (!(file >> temp->vw[j][k]))
                {
                    cout << "Error reading vw[" << j << "][" << k << "]" << endl;
                    file.close();
                    return;
                }
            }

            if (!(file >> temp->vb[j]))
            {
                cout << "Error reading vb[" << j << "]" << endl;
                file.close();
                return;
            }
        }
    }

    file.close();
    cout << "vwvb loaded successfully from vwvb.txt" << endl;

}
