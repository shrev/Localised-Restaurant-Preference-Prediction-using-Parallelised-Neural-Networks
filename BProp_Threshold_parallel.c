#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include<sys/time.h>

int numunits1=7,numunits2=7,numunits3=3,numpat=220;

double sigmoid(double x){

  return 1/(1+exp(-x));
}

void main() {
  
int i=0,j=0,k=0,p,epoch; double error=0.0;
  
  
double weight12[numunits1+1][numunits2+1],weight23[numunits2+1][numunits3+1];
double layer2In[numpat+1][numunits2+1],layer2Out[numpat+1][numunits2+1];
double layer3In[numpat+1][numunits3+1],layer3Out[numpat+1][numunits3+1];
double deltaO[numunits3+1],deltaH[numunits2+1],sumdow[numunits2+1];
double dw12[numunits1+1][numunits2+1],dw23[numunits2+1][numunits3+1];

double eta=0.7; 	/* gradient descent contribution */
double alpha=0.8; 	/* momentum term */
double smallwt=0.6; 	/* maximum absolute size of your initial weights */


double layer1Out[221][8] = {
{0,0,0,0,0,0,0,0},
{0,1,1,1,0,0,0,0},
{0,0,0,0,0,1,1,1},
{0,1,1,0,1,0,0,0},
{0,0,0,0,1,1,0,1},
{0,0,1,0,1,0,0,0},
{0,0,1,1,1,0,0,1},
{0,0,1,0,0,0,0,0},
{0,1,1,0,0,0,1,1},
{0,0,1,0,1,0,0,0},
{0,0,1,1,1,0,0,0},
{0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0},
{0,1,0,0,1,1,1,1},
{0,1,1,0,1,1,1,1},
{0,1,0,1,1,0,0,0},
{0,1,0,0,1,0,1,1},
{0,0,1,1,0,0,0,0},
{0,0,0,1,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,1,1,1,1,1,1,0},
{0,1,1,1,1,1,1,0},
{0,1,1,1,0,0,0,0},
{0,1,1,0,0,0,0,0},
{0,1,1,1,1,1,1,1},
{0,1,1,1,1,1,1,1},
{0,0,1,1,1,0,0,0},
{0,1,1,1,1,1,1,1},
{0,0,0,1,1,0,1,0},
{0,0,1,1,1,0,0,0},
{0,0,0,1,1,0,0,0},
{0,0,1,1,0,1,0,0},
{0,1,1,1,0,0,0,0},
{0,0,1,1,1,0,0,0},
{0,0,0,0,1,1,0,1},
{0,1,1,1,0,0,0,0},
{0,1,1,0,0,1,0,1},
{0,1,0,0,1,0,0,0},
{0,0,1,0,0,1,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,0,1,0,1,0},
{0,0,1,0,0,0,1,0},
{0,0,1,0,0,1,0,0},
{0,0,0,1,0,1,0,0},
{0,0,1,1,0,0,0,0},
{0,1,1,1,1,0,0,0},
{0,1,0,1,0,0,0,0},
{0,0,0,1,1,0,0,0},
{0,0,1,0,1,0,1,0},
{0,1,0,0,1,0,1,0},
{0,1,0,0,1,0,1,0},
{0,1,1,1,1,1,1,0},
{0,0,1,1,1,0,0,0},
{0,0,1,1,0,0,0,0},
{0,1,1,0,1,1,1,1},
{0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,1,1},
{0,0,1,0,1,0,1,0},
{0,1,1,0,0,0,0,0},
{0,1,1,0,0,0,0,0},
{0,1,1,0,1,0,0,0},
{0,0,0,1,1,0,0,1},
{0,1,1,1,1,0,1,0},
{0,1,0,0,1,0,0,0},
{0,0,1,0,0,1,0,1},
{0,1,1,1,1,1,1,0},
{0,1,1,0,1,0,0,0},
{0,1,0,0,1,0,0,0},
{0,1,1,0,0,0,0,0},
{0,1,0,1,1,0,0,0},
{0,1,0,1,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,0,0,1,0,1,0},
{0,0,0,0,0,0,1,1},
{0,0,1,1,1,0,0,0},
{0,1,1,0,1,0,1,0},
{0,0,1,0,1,1,0,1},
{0,1,0,0,1,0,1,0},
{0,1,1,0,0,0,0,0},
{0,1,1,0,1,0,1,0},
{0,0,1,1,1,0,1,0},
{0,0,1,0,1,0,0,0},
{0,1,1,1,0,0,1,0},
{0,1,0,0,1,0,1,0},
{0,0,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,1,1,0,1,0,0,1},
{0,1,0,0,1,0,0,1},
{0,1,1,1,1,0,0,0},
{0,0,1,1,1,0,0,0},
{0,1,1,1,1,0,0,0},
{0,0,1,1,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,0,1,0,1,0,0},
{0,0,1,1,1,0,0,0},
{0,1,1,1,1,0,0,1},
{0,1,1,1,1,1,1,1},
{0,1,1,1,1,1,1,1},
{0,1,1,0,1,0,0,0},
{0,0,1,1,0,0,0,0},
{0,0,1,0,1,1,0,1},
{0,1,1,0,1,0,0,0},
{0,1,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,0,1,0,0,1},
{0,0,0,0,1,1,0,0},
{0,0,0,0,1,1,0,0},
{0,0,1,0,1,0,0,1},
{0,1,1,0,1,0,1,1},
{0,1,1,0,1,0,0,1},
{0,1,0,0,1,1,1,1},
{0,0,1,0,1,1,1,1},
{0,1,1,0,1,0,0,0},
{0,1,1,0,1,0,0,0},
{0,1,1,0,1,0,0,0},
{0,0,1,0,0,1,0,0},
{0,0,1,0,1,0,0,0},
{0,1,1,1,0,0,1,0},
{0,1,1,1,1,1,1,0},
{0,1,1,0,0,0,1,1},
{0,0,1,1,0,0,0,0},
{0,0,0,0,1,1,0,0},
{0,1,1,0,0,1,0,1},
{0,1,0,0,1,0,0,0},
{0,1,1,1,0,1,0,0},
{0,1,0,0,1,0,0,0},
{0,1,1,0,1,0,0,0},
{0,0,0,1,0,0,0,1},
{0,1,1,1,1,1,1,0},
{0,0,1,1,0,0,0,0},
{0,1,1,1,1,1,1,1},
{0,0,1,1,0,1,0,0},
{0,1,0,0,1,0,0,1},
{0,0,1,0,1,1,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,1,1,1,1,0},
{0,0,1,1,1,0,1,0},
{0,0,1,0,1,0,0,0},
{0,0,1,0,0,1,0,0},
{0,0,1,1,1,0,0,0},
{0,1,0,1,1,0,0,1},
{0,1,0,1,1,0,0,1},
{0,0,1,0,1,0,0,0},
{0,1,1,0,1,0,0,1},
{0,0,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,1,1,1,1,1,1,1},
{0,1,1,0,0,0,0,0},
{0,1,0,0,1,0,0,0},
{0,0,0,0,1,1,0,0},
{0,1,1,0,1,0,0,0},
{0,0,1,1,0,0,0,1},
{0,0,0,1,1,0,0,0},
{0,1,0,0,1,0,0,0},
{0,1,0,0,1,0,0,0},
{0,1,1,0,1,1,1,0},
{0,1,1,0,1,0,1,1},
{0,1,1,0,0,1,0,1},
{0,1,1,0,1,1,1,0},
{0,0,1,1,0,0,1,1},
{0,1,1,0,1,0,1,0},
{0,1,0,1,1,0,0,1},
{0,0,1,0,1,0,0,0},
{0,1,1,0,0,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,1,1,0,1,0},
{0,1,0,0,0,0,1,1},
{0,0,1,1,1,0,0,1},
{0,0,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,1,1,0,0,1,0,1},
{0,1,1,1,0,1,1,1},
{0,1,1,1,0,1,1,1},
{0,0,0,0,1,0,0,1},
{0,0,1,0,1,1,0,1},
{0,1,1,0,0,0,1,1},
{0,1,1,1,1,1,1,0},
{0,0,1,1,0,0,0,1},
{0,0,1,0,0,0,1,0},
{0,1,1,0,0,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,0,0,1,0,1,0},
{0,0,1,1,0,0,0,0},
{0,0,0,1,1,0,1,0},
{0,1,0,0,1,0,0,1},
{0,1,0,0,1,1,0,0},
{0,0,1,0,0,1,0,0},
{0,0,0,1,0,1,0,1},
{0,0,1,1,0,0,0,0},
{0,0,1,1,0,0,0,0},
{0,0,1,0,1,0,0,1},
{0,1,1,0,0,0,1,1},
{0,1,1,0,1,0,0,1},
{0,1,1,0,0,1,0,1},
{0,1,0,0,0,0,1,0},
{0,1,1,0,1,0,1,1},
{0,1,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,1,0,1,0,1},
{0,0,1,0,1,0,0,1},
{0,0,1,1,0,0,0,0},
{0,1,1,0,0,1,0,1},
{0,0,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,1,1,0,0,0},
{0,1,1,0,0,0,0,0},
{0,0,1,0,1,0,0,0},
{0,0,1,1,1,0,1,0},
{0,0,1,1,0,0,0,0},
{0,1,1,0,1,1,0,0},
{0,0,1,0,1,0,0,0},
{0,1,1,0,1,0,0,0},
{0,0,1,0,1,0,0,0},
{0,1,1,0,0,1,0,1},
{0,1,0,1,0,1,1,1},
{0,0,1,1,0,0,0,0},
{0,1,1,0,0,0,0,0},
};


double target[221][4]= {
{0,0,0,0},
{0,1,0,0},
{0,0,0,1},
{0,0,1,0},
{0,0,0,1},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,0,0,1},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,0,0,1},
{0,1,0,0},
{0,0,0,1},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,0,0,1},
{0,0,0,1},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,0,0,1},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,0,1},
{0,1,0,0},
{0,0,0,1},
{0,0,0,1},
{0,1,0,0},
{0,0,1,0},
{0,0,0,1},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,0,0,1},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,0,1},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,0,0,1},
{0,0,0,1},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,0,0,1},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,0,1},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,0,0,1},
{0,0,0,1},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,0,0,1},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,0,1},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,0,0,1},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,0,0,1},
{0,0,0,1},
{0,0,0,1},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,0,1,0},
{0,0,0,1},
{0,0,0,1},
{0,0,0,1},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,0,1},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,1,0,0},
{0,1,0,0},
{0,1,0,0},
{0,0,0,1},
{0,0,1,0},
{0,0,1,0}
};


for(p=1;p<=numpat;p++) {
	for(i=1;i<=numunits1;i++) {
  		//printf("%lf\t",layer1Out[p][i]);
		}
	printf("\n");
}

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


struct timeval  TimeValue_Start;
struct timezone TimeZone_Start;
struct timeval  TimeValue_Final;
struct timezone TimeZone_Final;
long time_start, time_end;
double time_overhead;

gettimeofday(&TimeValue_Start, &TimeZone_Start);
 
for(epoch=0;epoch<=10000;epoch++){

	error=0.0;
	#pragma omp parallel for num_threads(4) reduction(+:error)
 	for(p=1;p<=numpat;p++) {  /* repeat for all the training patterns */

		for(j=1;j<=numunits2;j++) {    /* *Calculating values of hidden layer */
   		layer2In[p][j]=weight12[0][j];
    		for(i=1;i<=numunits1;i++){
      			layer2In[p][j]+=layer1Out[p][i]*weight12[i][j];
		}
    		layer2Out[p][j]=sigmoid(layer2In[p][j]);
		}

   
		for(k=1;k<=numunits3;k++) {   /* Calculating values of output layer*/
     		layer3In[p][k]=weight23[0][k];
     		for(j=1;j<=numunits2;j++) {
       			layer3In[p][k]+=layer2Out[p][j]*weight23[j][k];
      		}
     		layer3Out[p][k]=sigmoid(layer3In[p][k]);

    		error+=(double)(0.5*(target[p][k]-layer3Out[p][k])*(target[p][k]-layer3Out[p][k])); /* Compute Error */
     		//printf("%lf\n",error);
     		deltaO[k]=(target[p][k]-layer3Out[p][k])*layer3Out[p][k]*(1.0-layer3Out[p][k]); 
 		}

   		for(j=1;j<=numunits2;j++){  /* 'back-propagate' errors to hidden layer */
      		sumdow[j]=0.0;
      		for(k=1;k<=numunits3;k++) {
        		sumdow[j]+=weight23[j][k]*deltaO[k];
      		}
      		deltaH[j]=sumdow[j]+layer2Out[p][j]*(1.0-layer2Out[p][j]);
		}

   		for(j=1;j<=numunits2;j++) {  /* Update weights & biases : WeightIH */ 
		dw12[0][j]=(eta*deltaH[j])+(alpha*dw12[0][j]);
    		weight12[0][j]+=dw12[0][j];
    		for(i=1;i<=numunits1;i++) {

     			dw12[i][j]=(eta*layer1Out[p][i]*deltaH[j])+(alpha*dw12[i][j]);
      			weight12[i][j]+=dw12[i][j];
   		 }
  		 }

   		for(k=1;k<=numunits3;k++) {  /* Update weights & biases : WeightHO */ 

		dw23[0][k]=(eta*deltaO[k])+(alpha*dw23[0][k]);
     		weight23[0][k]+=dw23[0][k];
     		for(j=1;j<=numunits2;j++) {

        		dw23[j][k]=(eta*layer2Out[p][j]*deltaO[k])+(alpha*dw23[j][k]);
        		weight23[j][k]+=dw23[j][k];
     		}
   		}
   		
 	}
	if(error<50) break;
	printf("%dth iteration %lf\n",epoch,error);
}

gettimeofday(&TimeValue_Final, &TimeZone_Final);

time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
time_overhead = (time_end - time_start)/1000000.0;

/************************************************************************************************/

double test_layer1Out[31][8] = {
{0,1,0,0,1,0,1,1},
{0,1,0,0,1,0,0,0},
{0,0,1,1,0,0,0,1},
{0,0,0,0,0,1,1,0},
{0,0,1,1,0,0,0,1},
{0,0,0,0,0,1,1,0},
{0,0,0,0,1,1,1,1},
{0,0,0,0,1,1,1,1},
{0,0,0,0,1,1,1,1},
{0,0,1,1,0,0,0,1},
{0,1,0,0,1,0,0,0},
{0,1,0,1,0,0,0,1},
{0,1,1,1,1,0,1,0},
{0,1,1,0,0,1,0,1},
{0,0,0,1,1,0,0,0},
{0,0,1,1,0,0,0,0},
{0,1,1,0,0,1,0,0},
{0,0,0,1,1,0,0,0},
{0,0,0,0,0,1,1,0},
{0,0,0,0,0,1,1,0},
{0,0,0,1,0,0,1,0},
{0,0,0,1,0,0,1,0},
{0,0,0,0,0,1,1,0},
{0,1,1,1,1,0,0,1},
{0,1,1,0,1,0,0,1},
{0,0,1,1,0,0,0,0},
{0,1,1,1,0,0,0,0},
{0,0,0,0,0,1,1,1},
{0,1,1,0,1,0,0,0},
{0,0,0,0,1,1,0,1}
};

double test_target[31][4] = {
{0,1,0,0},
{0,0,0,1},
{0,1,0,0},
{0,0,0,1},
{0,1,0,0},
{0,0,0,1},
{0,0,0,1},
{0,0,0,1},
{0,0,0,1},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,1,0},
{0,0,0,1},
{0,0,1,0},
{0,1,0,0},
{0,0,0,1},
{0,1,0,0},
{0,0,0,1},
{0,1,0,0},
{0,1,0,0},
{0,0,1,0},
{0,0,0,1},
{0,0,1,0},
{0,1,0,0},
{0,0,1,0},
{0,1,0,0},
{0,0,0,1},
{0,0,1,0},
{0,0,0,1}
};

int test_numpat = 30;
double test_layer2In[test_numpat+1][numunits2+1],test_layer2Out[test_numpat+1][numunits2+1];
double test_layer3In[test_numpat+1][numunits3+1],test_layer3Out[test_numpat+1][numunits3+1];

for(p=1;p<=test_numpat;p++) {  /* repeat for all the training patterns */

		for(j=1;j<=numunits2;j++) {    /* *Calculating values of hidden layer */
   		test_layer2In[p][j]=weight12[0][j];
    		for(i=1;i<=numunits1;i++){
      			test_layer2In[p][j]+=test_layer1Out[p][i]*weight12[i][j];
		}
    		test_layer2Out[p][j]=sigmoid(test_layer2In[p][j]);
		}

   
		for(k=1;k<=numunits3;k++) {   /* Calculating values of output layer*/
     		test_layer3In[p][k]=weight23[0][k];
     		for(j=1;j<=numunits2;j++) {
       			test_layer3In[p][k]+=test_layer2Out[p][j]*weight23[j][k];
      		}
     		test_layer3Out[p][k]=sigmoid(test_layer3In[p][k]);
}}

    /************************************************************************************************/

double large; int index;
int count = 0;
for(p=1;p<=numpat;p++) {
	printf("%d Iteration -> \n",p);
	large = layer3Out[p][1]; index =1;

	for(k=1;k<=numunits3;k++){
  		printf("AV: %lf ; OV : %lf\n",target[p][k],layer3Out[p][k]);
		if(large<layer3Out[p][k]) { large = layer3Out[p][k]; index = k;}
		}

	if(target[p][index]==1) count++;
	printf("*********************************************************\n");
}

printf("\nTraining Accuracy for 220 : %lf\n", (count/220.0)*100);
printf("\nTime taken by Serial Code : %lf\n\n",time_overhead);

/************************************************************************************************/

double test_large, test_slarge; int test_index, test_sindex;
int test_count = 0;
for(p=1;p<=test_numpat;p++) {

	printf("%d Iteration -> \n",p);
	test_large = test_layer3Out[p][1]; test_index =1;
	test_slarge = test_layer3Out[p][1]; test_sindex =1;

	for(k=1;k<=numunits3;k++){
  		printf("AV: %lf ; OV : %lf\n",test_target[p][k],test_layer3Out[p][k]);
		if(test_large<test_layer3Out[p][k]) 
		{ test_sindex = index; test_slarge = test_large; test_large = test_layer3Out[p][k]; test_index = k;} 
		else if(test_slarge<test_layer3Out[p][k]) 
		{ test_slarge = test_layer3Out[p][k]; test_sindex = k;}
		}
	

	if(test_target[p][test_index]==1) test_count++;
else 	if(test_target[p][test_sindex]==1 && test_large-test_slarge<=0.1) test_count++;

	printf("*********************************************************\n");
}

printf("\nTraining Accuracy for 220 : %lf\n", (count/220.0)*100);
printf("\nTime taken by Serial Code : %lf\n",time_overhead);
printf("\nTesting Accuracy for 30   : %lf\n\n", (test_count/30.0)*100);
}