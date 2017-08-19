#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include<sys/time.h>

int numunits1=2,numunits2=2,numunits3=1,numpat=4;

double sigmoid(double x)
{
  return 1/1+exp(-x);
}

void main()
{
  int i=0,j=0,k=0,p,epoch;
  double error=0.0;
  
  
double target[numpat][numunits3];
double weight12[numunits1+1][numunits2+1],weight23[numunits2+1][numunits3+1];
double layer1Out[numpat+1][numunits1+1],layer2In[numpat+1][numunits2+1],layer2Out[numpat+1][numunits2+1],layer3In[numpat+1][numunits3+1],layer3Out[numpat+1][numunits3+1];
double  somdow[numunits2+1],deltaO[numunits3+1],deltaH[numunits2+1],dw12[numunits1+1][numunits2+1],dw23[numunits2+1][numunits3+1],sumdow[numunits2+1];
double eta=0.1,alpha=0.3,smallwt=0.5;

layer1Out[1][1]=0.0;
layer1Out[1][2]=0.0;
layer1Out[2][1]=0.0;
layer1Out[2][2]=1.0;
layer1Out[3][1]=1.0;
layer1Out[3][2]=0.0;
layer1Out[4][1]=1.0;
layer1Out[4][2]=1.0;

target[1][1]=0.0;
target[2][1]=1.0;
target[3][1]=1.0;
target[4][1]=1.0;

  for( j = 1 ; j <= numunits2 ; j++ ) { /* initialize WeightIH and DeltaWeightIH */
for( i = 0 ; i <= numunits1 ; i++ ) {
dw12[i][j] = 0.0 ;
weight12[i][j] = 2.0 * ( ((double)rand()/(double)(RAND_MAX+1)) - 0.5 ) * smallwt ;
}
}
for( k = 1 ; k <= numunits3 ; k ++ ) { /* initialize WeightHO and DeltaWeightHO */
for( j = 0 ; j <= numunits2; j++ ) {
dw23[j][k] = 0.0 ;
weight23[j][k] = 2.0 * ( ((double)rand()/((double)RAND_MAX+1) - 0.5 )) * smallwt ;
}
printf("%lf",(double)rand()/(double)RAND_MAX);
}

  //calculating value of hidden layer
for(epoch=0;epoch<=100;epoch++)
{
for(p=1;p<=numpat;p++)
{
  for(j=1;j<=numunits2;j++)
  {
    layer2In[p][j]=weight12[0][j];
    for(i=1;i<=numunits1;i++)
    {
      layer2In[p][j]+=layer1Out[p][i]*weight12[i][j];
      
    }
    //printf("%lf\n",layer2In[p][j]);
    layer2Out[p][j]=sigmoid(layer2In[p][j]);
    //printf("%lf\n",layer2Out[p][j]);
   }
   //calculating values of output layer
   for(k=1;k<=numunits3;k++)
   {
     layer3In[p][k]=weight23[0][k];
     for(j=1;j<=numunits2;j++)
     {
       layer3In[p][k]+=layer2Out[p][j]*weight23[j][k];
       }
     layer3Out[p][k]=sigmoid(layer3In[p][k]);
     error+=(double)(0.5*(target[p][k]-layer3Out[p][k])*(target[p][k]-layer3Out[p][k]));
   //  printf("%lf %lf ",layer3Out[p][k],target[p][k]);
	// printf("%lf\n",error);
     //sum squared error
     deltaO[k]=(target[p][k]-layer3Out[p][k])*layer3Out[p][k]*(1.0-layer3Out[p][k]);
   }
   for(j=1;j<=numunits2;j++)
   {
      sumdow[j]=0.0;
      for(k=1;k<=numunits3;k++)
      {
        sumdow[j]+=weight23[j][k]*deltaO[k];
      }
      deltaH[j]=sumdow[j]+layer2Out[p][j]*(1.0-layer2Out[p][j]);
   }

   for(j=1;j<=numunits2;j++)
   {
    dw12[0][j]=eta*deltaH[j]+alpha*dw12[0][j];
    weight12[0][j]+=dw12[0][j];
    for(i=1;i<=numunits1;i++)
    {
      dw12[i][j]=eta*layer1Out[p][i]*deltaH[j]+alpha*dw12[i][j];
      weight12[i][j]+=dw12[i][j];//update layer1-layer2 weights
    }
   }

   for(k=1;k<=numunits3;k++)
   {
     dw23[0][k]=eta*deltaO[k]+alpha*dw23[0][k];
     weight23[0][k]+=dw23[0][k];
     for(j=1;j<=numunits2;j++)
     {
        dw23[j][k]=eta*layer2Out[p][j]*deltaO[k]+alpha*dw23[j][k];
        weight23[j][k]+=dw23[j][k];
     }
   }
 }

if(error<0.0005) break;
//printf("%dth iteration %lf\n",epoch,error);
}




for(p=1;p<=numpat;p++)
{
for(k=1;k<=numunits3;k++)
{
  printf("value %d\n",layer3Out[p][k]);
}
}
}
