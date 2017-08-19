#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include<sys/time.h>

int numunits1=2,numunits2=2,numunits3=1,numpat=4;

double sigmoid(double x){
  return 1/(1+exp(-x));
}

double sign(double x){
  return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

void main() {
  
int i=0,j=0,k=0,p,epoch; double error=0.0;
  
  
double target[numpat][numunits3];
double weight12[numunits1+1][numunits2+1],weight23[numunits2+1][numunits3+1];
double layer1Out[numpat+1][numunits1+1];
double layer2In[numpat+1][numunits2+1],layer2Out[numpat+1][numunits2+1];
double layer3In[numpat+1][numunits3+1],layer3Out[numpat+1][numunits3+1];
double deltaO[numunits3+1],deltaH[numunits2+1],sumdow[numunits2+1];
double dw12[numunits1+1][numunits2+1],dw23[numunits2+1][numunits3+1];

double deltaMaximum = 50, deltaMinimum = 0.000001;
double positiveEta=1.2; 
double negativeEta=0.5; /* gradient descent contribution */
double alpha=0.9; 	/* momentum term */
double smallwt=0.6; 	/* maximum absolute size of your initial weights */

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

for(p=1;p<=numpat;p++) {
	for(i=1;i<=numunits1;i++) {
  		printf("%lf\t",layer1Out[p][i]);
		}
	printf("\n");
}

for( j = 1 ; j <= numunits2 ; j++ ) { /* initialize WeightIH and DeltaWeightIH */
	for( i = 0 ; i <= numunits1 ; i++ ) {
		dw12[i][j] = 0.0 ;
		weight12[i][j] = 2.0 * ( ((double)rand()/((double)RAND_MAX+1) - 0.5 )) * smallwt ;
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
 
for(epoch=0;epoch<=10;epoch++){

	error=0.0;
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
     		printf("%lf\n",error);
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

		/* Update biases WeightIH */ 
		double bias_prevGradient = 0;
		double bias_currentGradient =deltaH[i] ;		
		double bias_signChange = sign(bias_currentGradient * bias_prevGradient);
    		double biasChange = 0;


			double bias_prevUpdateValue = dw12[0][j];
			double bias_newUpdateValue = bias_prevUpdateValue;

			if(bias_signChange > 0) {
			   bias_newUpdateValue = positiveEta * bias_prevUpdateValue;
			   if(bias_newUpdateValue > deltaMaximum) bias_newUpdateValue = deltaMaximum;
			   biasChange = -1*sign(bias_currentGradient)*bias_newUpdateValue;
                	 } else 
			if (bias_signChange < 0){
        		   bias_newUpdateValue = negativeEta * bias_prevUpdateValue;
		           if(bias_newUpdateValue < deltaMaximum) bias_newUpdateValue = deltaMinimum;
        		   biasChange = -bias_prevUpdateValue;
			   bias_currentGradient = 0;
    			 } else {
        		   bias_newUpdateValue = bias_prevUpdateValue;
          		   biasChange = 0;
    			}
      			      			
			dw12[0][j] = biasChange;			
			weight12[0][j]+=biasChange;
			bias_prevGradient = bias_currentGradient;


		/* Update weights WeightIH */ 
		double prevGradient = 0;
		double currentGradient =deltaH[i] ;		
		double signChange = sign(currentGradient * prevGradient);
    		double weightChange = 0;

    	  	for(i=1;i<=numunits1;i++) {

			double prevUpdateValue = dw12[i][j];
			double newUpdateValue = prevUpdateValue;

			if(signChange > 0) {
			   newUpdateValue = positiveEta * prevUpdateValue;
			   if(newUpdateValue > deltaMaximum) newUpdateValue = deltaMaximum;
			   weightChange = -1*sign(currentGradient)*newUpdateValue;
                	 } else 
			if (signChange < 0){
        		   bias_newUpdateValue = negativeEta * bias_prevUpdateValue;
		           if(bias_newUpdateValue < deltaMaximum) bias_newUpdateValue = deltaMinimum;
        		   biasChange = -bias_prevUpdateValue;
			   bias_currentGradient = 0;
    			 } else {
        		   newUpdateValue = prevUpdateValue;
          		   weightChange = 0;
    			}
      			 
			dw12[i][j] = weightChange;     			
			weight12[i][j]+=weightChange;
			prevGradient = currentGradient;


			}// end of for
   		}// end of Update weights & biases : WeightIH */ 

		/***************************************************************************************************/

   		for(j=1;j<=numunits2;j++) {  /* Update weights & biases : WeightHO */ 

		/* Update biases WeightHO*/ 
		double bias_prevGradient = 0;
		double bias_currentGradient =deltaH[i] ;		
		double bias_signChange = sign(bias_currentGradient * bias_prevGradient);
    		double biasChange = 0;


			double bias_prevUpdateValue = dw23[0][j];
			double bias_newUpdateValue = bias_prevUpdateValue;

			if(bias_signChange > 0) {
			   bias_newUpdateValue = positiveEta * bias_prevUpdateValue;
			   if(bias_newUpdateValue > deltaMaximum) bias_newUpdateValue = deltaMaximum;
			   biasChange = -1*sign(bias_currentGradient)*bias_newUpdateValue;
                	 } else 
			if (bias_signChange < 0){
        		   bias_newUpdateValue = negativeEta * bias_prevUpdateValue;
		           if(bias_newUpdateValue < deltaMaximum) bias_newUpdateValue = deltaMinimum;
        		   biasChange = -bias_prevUpdateValue;
			   bias_currentGradient = 0;
    			 } else {
        		   bias_newUpdateValue = bias_prevUpdateValue;
          		   biasChange = 0;
    			}
      			      			
			dw23[0][j] = biasChange;			
			weight23[0][j]+=biasChange;
			bias_prevGradient = bias_currentGradient;


		/* Update weights WeightHO */ 
		double prevGradient = 0;
		double currentGradient =deltaH[i] ;		
		double signChange = sign(currentGradient * prevGradient);
    		double weightChange = 0;

    	  	for(i=1;i<=numunits1;i++) {

			double prevUpdateValue = dw23[i][j];
			double newUpdateValue = prevUpdateValue;

			if(signChange > 0) {
			   newUpdateValue = positiveEta * prevUpdateValue;
			   if(newUpdateValue > deltaMaximum) newUpdateValue = deltaMaximum;
			   weightChange = -1*sign(currentGradient)*newUpdateValue;
                	 } else 
			if (signChange < 0){
        		   newUpdateValue = negativeEta * prevUpdateValue;
		           if(newUpdateValue < deltaMaximum) newUpdateValue = deltaMinimum;
        		   weightChange = -prevUpdateValue;
			   currentGradient = 0;
    			 } else {
        		   newUpdateValue = prevUpdateValue;
          		   weightChange = 0;
    			}
      			 
			dw23[i][j] = weightChange;     			
			weight23[i][j]+=weightChange;
			prevGradient = currentGradient;


			}// end of for
   		}// end of Update weights & biases : WeightHO 
		
		/*******************************************************************************************************/

   		
 	}
	if(error<0.0000) break;
	printf("%dth iteration %lf\n",epoch,error);
}

gettimeofday(&TimeValue_Final, &TimeZone_Final);

time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
time_overhead = (time_end - time_start)/1000000.0;

printf("\nTime taken by Serial = %lf\n\n",time_overhead);


for(p=1;p<=numpat;p++) {
	for(k=1;k<=numunits3;k++){
  		printf("value %lf\n",layer3Out[p][k]);
		}
	}
}
