
#include "stdafx.h"
#include<stdio.h>
#include<conio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

double sn[500000];							//stores samples of training signals
unsigned int count1 = 0;					//stores no. of samples in training signal

int start;									//start pointer of frame
int mid;									//mid pointer of frame
int end;									//end marker of frame
double sil[500000];							//stores samples of silence frame
unsigned int count2 = 0;					//stores no. of samples in silence signal
int p=12;
int q=12;
int y,z;

double ip[500000];							//stores samples of testing signal
unsigned int count4=0;						//stores no. of samples of tokhura file
unsigned int count5=0;						//stores no. of samples of raised sine window file
unsigned int count6=0;						//stores no. of samples of testing signal
double r[15];								//stores ri's
double a[15];								//stores ai's
double c[15];								//stores ci's
double w[15];								//stores Raised Sine Window samples
double f[5000][13];							//stores ci's of the training frames
double v[13];								//stores tokhura weights
double dtcep[50];							//stores distance of one testing signal from all training signals
int minindex=1;								//stores index of minimum distance
double ct[15];								//stores ci's of one testing signal
double* tn;
unsigned int count=0;						//stores count for correct outputs
double accuracy=0;							//stores accuracy of system
unsigned int count7=0;
unsigned int count8=0;

#define M 32								//Number of observation symbols
#define N 5									//Number of states
#define Tj 5000								//Length of Observation Sequence
#define limit 1.01
int T=0;

long double pi[N+1] = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
long double A[N+1][N+1]={0};
long double B[N+1][M+1]={0};
long double pinew[N+1]={0};
long double Anew[N+1][N+1]={0};
long double Bnew[N+1][M+1]={0};
long double Mpi[50][N+1]={0};
long double MA[50][N+1][N+1]={0};
long double MB[50][N+1][M+1]={0};

long double Alpha[N+1][Tj+1]={0};
long double Beta[N+1][Tj+1]={0};
long double Gamma[N+1][Tj+1]={0};
long double Delta[N+1][Tj+1]={0};
int Psi[N+1][Tj+1]={0};
long double Xi[N+1][N+1][Tj+1]={0};

long double P[50];
long double Pstar=0;
int qtemp=0;
int qstar[5000]={0};							//State sequence
int O[Tj+1]={0};								//Observation sequence
double cb[33][13]={0};							//Codebook
int loop=1;


void dcshift(double* a,int b)					//shifts the signal by calculating dc shift 
{
	long double avg=0;
	for(int i=0;i<count2;i++)
	{
		avg+=sil[i];
	}
	avg=avg/count2;
	for(int i=0;i<b;i++)
	{
		a[i]-=avg;
	}
}

void normalize(double* a,int b)					//normalizes the signal
{
	double l=a[0];
	for(int i=0;i<b;i++)
	{
		if(a[i]>l)
		{
			l=a[i];
		}
	}

	for(int i=0;i<b;i++)
	{
		double k;
		k=a[i];
		k*=5000;
		k=(k/l);
		a[i]=k;
	}
}

void steady(double* a,int b)				//finds steady state of signal
{
	double x[500]={0};
	int t=b/100;
	int i=0;
	int j=0;
	
	//printf("total frames->%d",t);
	while(t--)
	{
		long int u[100];
		long int enr=0;
		for(int q=0;i<b && q<100;i++,q++)
		{
			u[q]=a[i]*a[i];
		}
	
		for(int i=0;i<100;i++)
		{
			enr+=u[i];
		}
		enr=enr/100;
		x[j++]=enr;
	}

	t=b/100;
	double avg=0;
	for(int i=0;i<t;i++)
	{
		double temp=x[i]/t;
		avg+=temp;
	}

	for(int i=0;i<t;i++)
	{
		if(x[i]>(limit)*avg)
		{
			start=i*100;
			break;
		}
	}
	//printf("start - %d\n",start);
	for(j=t-1;j>0;j--)
	{
		if(x[j]>(limit)*avg)
		{
			end=j*100;
			break;
		}
	}
	//printf("end - %d\n",end);
}

void calculate_ri(double* sn,double* r)			//calculates ri's from signal
{
 	for(int i=0;i<=p;i++)
 	{
 		r[i]=0;
 		for(int j=0;j<=319-i;j++)
 		{	
 			r[i]+=sn[j]*sn[j+i];
		}
	}
	//printf("\n**********Ri's'**********\n");
 	for(int i=0;i<=p;i++)
 	{
 		//printf("%lf\n",r[i]);
	}
}

void durbin(double* r,double* a)			//calculates ai's from ri's
{
	double e[500];
	double k[500];
	double x[15][15];
	int i,j;
	double t;

	e[0]=r[0];

	for(i=1;i<=p;i++)
	{
		t=0;
		if(i!=1)
		{
			for(j=1;j<=(i-1);j++)
			{
				t+=x[j][i-1]*r[i-j];
			}
		}
		k[i]=(r[i]-t)/e[i-1];
		x[i][i]=k[i];	
		
		for(j=1;j<=(i-1);j++)
		{
			x[j][i]=x[j][i-1]-k[i]*x[i-j][i-1];
		}
		e[i]=(1-k[i]*k[i])*e[i-1];
	}
	//printf("\n**********Ai's**********\n");	
	for(int i=1;i<=p;i++)
	{
		a[i]=x[i][p];
		//printf("%lf\n",a[i]);
	}
}

double energy1(double *a)		//calculates energy of signal
{
	double enr=0;
	for(int i=0;i<320;i++)
	{
		enr+=(a[i]*a[i])/320;
	}
	return enr;
}

void cepstral(double* a,double* c)			//calculates ci's from ai's
{
	double t;
	//a[0]=log(energy1(tn));
	for(int i=1;i<=p;i++)
	{
		t=0;
		for(int j=1;j<=i-1;j++)
		{
			t+=((double)j/(double)i)*c[j]*a[i-j];
		}
		c[i]=a[i]+t;
		//c[i]=c[i]*w[i];
	}
	/*
	printf("\n**********Ci's**********\n");
	for(int i=1;i<=p;i++)
 	{
 		printf("%lf\n",c[i]);
	}
	printf("\n");
	*/
}

void tokhura(double ct[5000][13])				//calculates tokhura distance
{
	for(int iter=1;iter<=T;iter++)
	{
		for(int i=1;i<=32;i++)
		{
			double k=0;
			for(int j=1;j<=12;j++)
			{
				double g=(ct[iter][j]-cb[i][j])*(ct[iter][j]-cb[i][j]);
				k+=v[j]*g;
			}
			dtcep[i]=k;
		}

		double min=dtcep[1];
		minindex=1;
		for(int i=2;i<=32;i++)
		{
			if(dtcep[i]<min)
			{
				min=dtcep[i];
				minindex=i;
			}
		}
		O[iter]=minindex;
	}
}

void Forward()								//Calculating Alpha matrix
{
	long double temp=0;
	long double Prob=0;

	for(int i=1;i<=N;i++)
	{
		Alpha[i][1]=pi[i]*B[i][O[1]];
	}
	
	for(int t=1;t<=T-1;t++)
	{
		for(int j=1;j<=N;j++)
		{
			temp=0;
			for(int i=1;i<=N;i++)
			{
				temp+=Alpha[i][t]*A[i][j];
			}
			temp*=B[j][O[t+1]];
			Alpha[j][t+1]=temp;
		}
	}
	
	for(int i=1;i<=N;i++)
	{
		Prob+=Alpha[i][T];					//Probability of observation sequence given the model
	}
	//printf("%Le \n",Prob);
}

void Backward()								//Calculating Beta matrix
{
	for(int i=1;i<=N;i++)
	{
		Beta[i][T]=1;
	}

	for(int t=T-1;t>=1;t--)
	{
		for(int i=1;i<=N;i++)
		{
			long double temp=0;
			for(int j=1;j<=N;j++)
			{
				temp+=A[i][j]*B[j][O[t+1]]*Beta[j][t+1];
			}
			Beta[i][t]=temp;
		}
	}
}

void GammaVar()								//Calculating Gamma matrix
{
	for(int t=1;t<=T;t++)
	{
		for(int j=1;j<=N;j++)
		{
			long double temp1=Alpha[j][t]*Beta[j][t];
			long double temp2=0;
			for(int k=1;k<=N;k++)
			{
				temp2+=Alpha[k][t]*Beta[k][t];
			}
			Gamma[j][t]=temp1/temp2;
		}
	}
}

void Viterbi()									//Calculating P* and state sequence
{
	for(int i=1;i<=N;i++)
	{	
		Delta[i][1]=pi[i]*B[i][O[1]];
		Psi[i][1]=0;
	}

	for(int t=2;t<=T;t++)
	{
		for(int j=1;j<=N;j++)
		{
			long double max1=0,temp1=0;
			int temp2=0;
			for(int i=1;i<=N;i++)
			{
				temp1=Delta[i][t-1]*A[i][j];
				if(temp1>=max1)
				{
					max1=temp1;
					temp2=i;
				}
			}
			Delta[j][t]=max1*B[j][O[t]];
			Psi[j][t]=temp2;
		}
	}

	long double max2=0;
	for(int i=1;i<=N;i++)
	{
		long double temp3=Delta[i][T];
		if(temp3>=max2)
		{
			max2=temp3;
			qstar[T]=i;
		}
		Pstar=max2;
		qtemp=i;
	}
	
	for(int t=T-1;t>=1;t--)
	{
		int temp4=qstar[t+1];
		qstar[t]=Psi[temp4][t+1];
	}
}

void Reestimation()								//Adjusting Model Parameters
{
	for(int t=1;t<=T;t++)
	{
		for(int i=1;i<=N;i++)
		{
			for(int j=1;j<=N;j++)
			{
				long double temp1=Alpha[i][t]*A[i][j]*B[j][O[t+1]]*Beta[j][t+1];
				long double temp2=0;
				for(int p=1;p<=N;p++)
				{
					for(int q=1;q<=N;q++)
					{
						temp2+=Alpha[p][t]*A[p][q]*B[q][O[t+1]]*Beta[q][t+1];
					}
				}
				Xi[i][j][t]=temp1/temp2;
			}
		}
	}

	for(int i=1;i<=N;i++)
	{
		pi[i]=Gamma[i][1];						//pinew
	}

	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=N;j++)
		{
			long double temp1=0,temp2=0;
			for(int t=0;t<=T-1;t++)
			{
				temp1+=Xi[i][j][t];
				temp2+=Gamma[i][t];
			}
			A[i][j]=temp1/temp2;				//Anew
		}
	}

	for(int j=1;j<=N;j++)
	{
		for(int k=1;k<=M;k++)
		{
			B[j][k] = 0;						//Bnew
		}

		long double temp2=0;
		for(int t=1;t<=T;t++)
		{
			B[j][O[t]]+=Gamma[j][t];			//Bnew
			temp2+=Gamma[j][t];
		}

		for(int i=1;i<=M;i++)
		{
			B[j][i]/=temp2;						//Bnew
		}
	}
}

void printstates()								//Print P* and state sequence for each iteration
{
	printf("Pstar: %Le\n",Pstar);
	
	printf("State Sequence: ");
	for(int i=1;i<=T;i++)
	{
		printf("%d ",qstar[i]);
	}
	printf("\n");
}

void load_inertia_model()								//Loads model and obs sequences from input files
{
	//pi[N+1] = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0};

	FILE *f1=fopen("A1_matrix.txt","r");
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=N;j++)
		{
			fscanf(f1,"%Lf",&A[i][j]);
		}
	}
	fclose(f1);

	FILE *f2=fopen("B1_matrix.txt","r");
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=M;j++)
		{
			fscanf(f1,"%Lf",&B[i][j]);
		}
	}
	fclose(f2);
}

void load_average_model(int i,int p)								//Loads average model from input files
{
	//pi[] = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0};

	char filename[160];
	sprintf(filename,"models/name%d_A%d_matrix.txt",i,p);
	FILE *f1 = fopen(filename, "r");
	//printf("%s\n",filename);
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=N;j++)
		{
			fscanf(f1,"%Le",&A[i][j]);
		}
	}
	fclose(f1);

	char filename2[160];
	sprintf(filename2,"models/name%d_B%d_matrix.txt",i,p);
	FILE *f2 = fopen(filename2, "r");
	//printf("%s\n",filename2);
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=M;j++)
		{
			fscanf(f2,"%Le",&B[i][j]);
		}
	}
	fclose(f2);
}

void nullify_model()					//reset model(pi,A,B) to 0
{
	for(int i=1;i<=N;i++)
	{
		pinew[i]=0;
	}
	
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=N;j++)
		{
			Anew[i][j]=0;
		}
	}
	
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=M;j++)
		{
			Bnew[i][j]=0;
		}
	}
}

void nullify_modellive()					//reset model(pi,A,B) to 0
{
	for(int i=1;i<=N;i++)
	{
		pi[i]=0;
	}
	
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=N;j++)
		{
			A[i][j]=0;
		}
	}
	
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=M;j++)
		{
			B[i][j]=0;
		}
	}
}

void savemodel(int k,int i)				//saves model to files
{
	char filename[160];
	sprintf(filename,"models/name%d_A%d_matrix.txt",k,(i+1));
	FILE *f1 = fopen(filename, "w");			//wb
	printf("%s model written\n",filename);
	for(int i=1; i<=N; i++)
	{
		for(int j=1; j<=N; j++)
		{
			fprintf(f1, "%Le\t",MA[k][i][j]);
		}
		fprintf(f1,"\n");
	}
	fclose(f1);

	char filename2[160];
	sprintf(filename2,"models/name%d_B%d_matrix.txt",k,(i+1));
	FILE *f2 = fopen(filename2, "w");
	printf("%s model written\n",filename2);
	for(int i=1; i<=N; i++)
	{
		for(int j=1; j<=M; j++)
		{
			fprintf(f2,"%Le\t", MB[k][i][j]);
		}
		fprintf(f2, "\n");
	}
	fclose(f2);
}

void savemodellive(int k)				//saves model to files
{
	char filename[160];
	sprintf(filename,"models/name%d_A%d_matrix.txt",k,4);
	FILE *f1 = fopen(filename, "w");			//wb
	printf("%s model written\n",filename);
	for(int i=1; i<=N; i++)
	{
		for(int j=1; j<=N; j++)
		{
			fprintf(f1, "%Le\t",A[i][j]);
		}
		fprintf(f1,"\n");
	}
	fclose(f1);

	char filename2[160];
	sprintf(filename2,"models/name%d_B%d_matrix.txt",k,4);
	FILE *f2 = fopen(filename2, "w");
	printf("%s model written\n",filename2);
	for(int i=1; i<=N; i++)
	{
		for(int j=1; j<=M; j++)
		{
			fprintf(f2,"%Le\t", B[i][j]);
		}
		fprintf(f2, "\n");
	}
	fclose(f2);
}

void fixB()								//Fixes B Matrix (change zeros to thresholds)
{
	for(int i=1;i<=N;i++)
	{
		int count=0;
		int max=-1;
		int maxindex=-1;
		int sum=0;
		
		for(int j=1;j<=M;j++)
		{
			if(max<B[i][j])
			{
				max=B[i][j];
				maxindex=j;
			}
			if(B[i][j]<pow(10.0,-30))
			{
				count++;
				sum+=B[i][j];
				B[i][j]=pow(10.0,-30);
			}	
		}
		B[i][maxindex]-=count*pow(10.0,-30)+sum;
	}
}

void printmodel()				//prints model(pi,A,B)
{
	printf("\nPi\n");
	for(int i=1;i<=N;i++)
	{
		printf("%Le ",pinew[i]);
	}
	printf("\n");

	printf("\nA Matrix\n");
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=N;j++)
		{
			printf("%Le ",Anew[i][j]);
		}
		printf("\n");
	}
	
	printf("\nB Matrix\n");
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=M;j++)
		{
			printf("%Le ",Bnew[i][j]);
		}
		printf("\n\n");
	}
}

void fun()					//gives model for each utterance
{
	for(int i=1;i<=20;i++)
	{	
		//printf("Iteration %d\n",i);
		Forward();
		Backward();
		GammaVar();
		Viterbi();
		Reestimation();
		fixB();
		//printstates();
		//printf("\n_____________________\n");
	}
	fixB();

	for(int i=1;i<=N;i++)
	{
		pinew[i]+=pi[i];
	}
	
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=N;j++)
		{
			Anew[i][j]+=A[i][j];
		}
	}
	
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=M;j++)
		{
			Bnew[i][j]+=B[i][j];
		}
	}
}

void funlive()
{
	for(int i=1;i<=20;i++)
	{	
		//printf("Iteration %d\n",i);
		Forward();
		Backward();
		GammaVar();
		Viterbi();
		Reestimation();
		fixB();
		//printstates();
		//printf("\n_____________________\n");
	}
	fixB();
}

void average(int k)			//gives model for each digit
{
	for(int i=1;i<=N;i++)
	{
		pinew[i]/=30;
	}
	
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=N;j++)
		{
			Anew[i][j]/=30;
		}
	}
	
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=M;j++)
		{
			Bnew[i][j]/=30;
		}
	}
	
	for(int i=1;i<=N;i++)
	{
		Mpi[k][i]=pinew[i];
	}
	
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=N;j++)
		{
			MA[k][i][j]=Anew[i][j];
		}
	}
	
	for(int i=1;i<=N;i++)
	{
		for(int j=1;j<=M;j++)
		{
			MB[k][i][j]=Bnew[i][j];
		}
	}

}

void test_forward(int i)								//Calculating Alpha matrix and outputs the spoken digit by calculating P(O/model)
{
	for(int k=1;k<=15;k++)
	{
		long double temp=0;
		long double Prob=0;

		for(int i=1;i<=N;i++)
		{
			Alpha[i][1]=Mpi[k][i]*MB[k][i][O[1]];
		}
	
		for(int t=1;t<=T-1;t++)
		{
			for(int j=1;j<=N;j++)
			{
				temp=0;
				for(int i=1;i<=N;i++)
				{
					temp+=Alpha[i][t]*MA[k][i][j];
				}
				temp*=MB[k][j][O[t+1]];
				Alpha[j][t+1]=temp;
			}
		}
	
		for(int i=1;i<=N;i++)
		{
			Prob+=Alpha[i][T];					//Probability of observation sequence given the model
		}
		P[k]=Prob;
		//printf("%Le \n",Prob);
	}

	long double max=P[1];
	int maxindex=1;
	for(int i=2;i<=15;i++)
	{
		if(P[i]>max)
		{
			max=P[i];
			maxindex=i;
		}
	}
	printf("%d ",maxindex);
	
	if(maxindex==i)
	{
		count++;
	}
}

void print(int a)
{
	switch(a)
	{
		case 1:printf("\nWord recognized as Old\nSynonyms for Old are : Aged, Ancient, Mature\n");break;
		case 2:printf("\nWord recognized as New\nSynonyms for New are : Advanced, Modern, Recent\n");break;
		case 3:printf("\nWord recognized as Dead\nSynonyms for Dead are : Buried, Late, Deceased\n");break;	
		case 4:printf("\nWord recognized as Music\nSynonyms for Music are : Melody, Tune, Opera\n");break;
		case 5:printf("\nWord recognized as Learn\nSynonyms for Learn are : Grasp, Study, Read\n");break;
		case 6:printf("\nWord recognized as Expand\nSynonyms for Expand are : Grow, Increase, Enlarge\n");break;
		case 7:printf("\nWord recognized as Search\nSynonyms for Search are : Explore, Examine, Inspect\n");break;
		case 8:printf("\nWord recognized as Problem\nSynonyms for Problem are : Complication, Dispute, Obstacle\n");break;
		case 9:printf("\nWord recognized as Start\nSynonyms for Start are : Dawn, Kickoff, Open\n");break;
		case 10:printf("\nWord recognized as Fight\nSynonyms for Fight are : Altercation, Battle, Brawl\n");break;
		case 11:printf("\nWord recognized as Friend\nSynonyms for Friend are : Buddy, Colleague, Campanion\n");break;
		case 12:printf("\nWord recognized as Portal\nSynonyms for Portal are : Doorway, Entrance, Gateway\n");break;
		case 13:printf("\nWord recognized as Give\nSynonyms for Give are : Award, Donate, Present\n");break;
		case 14:printf("\nWord recognized as Speak\nSynonyms for Speak are : Communicate, Convey, Chat\n");break;
		case 15:printf("\nWord recognized as Rocket\nSynonyms for Rocket are : Missile, Spacecraft, Booster\n");break;
	}
}

void test_forwardlive()								//Calculating Alpha matrix and outputs the spoken digit by calculating P(O/model)
{
	for(int k=1;k<=15;k++)
	{
		long double temp=0;
		long double Prob=0;

		for(int i=1;i<=N;i++)
		{
			Alpha[i][1]=Mpi[k][i]*MB[k][i][O[1]];
		}
	
		for(int t=1;t<=T-1;t++)
		{
			for(int j=1;j<=N;j++)
			{
				temp=0;
				for(int i=1;i<=N;i++)
				{
					temp+=Alpha[i][t]*MA[k][i][j];
				}
				temp*=MB[k][j][O[t+1]];
				Alpha[j][t+1]=temp;
			}
		}
	
		for(int i=1;i<=N;i++)
		{
			Prob+=Alpha[i][T];					//Probability of observation sequence given the model
		}
		P[k]=Prob;
		//printf("%Le \n",Prob);
	}

	long double max=P[1];
	int maxindex=1;
	for(int i=2;i<=15;i++)
	{
		if(P[i]>max)
		{
			max=P[i];
			maxindex=i;
		}
	}
	print(maxindex);
	
}

void readfiles()				//load some necessary files
{
	FILE *f2 = fopen("silence.txt","r");		//opening silence file
	if (f2 != NULL)
	{
    	while (!feof(f2))
		{
       		fscanf(f2,"%lf",&sil[count2]);
       		count2++;
    	}
 	}
 	else
 	printf("\nSilence File cannot be opened");
 	fclose(f2);
 	
 	FILE *f4 = fopen("Tokhura_weights.txt","r");		//opening tokhura weights file
	if (f4 != NULL)
	{
    	while (!feof(f4))
		{
       		fscanf(f4,"%lf",&v[count4]);
       		count4++;
    	}
 	}
 	else
 	printf("\nTokhura_weights File cannot be opened");
 	fclose(f4);
 	
 	FILE *f5 = fopen("Raised_sine_window.txt","r");		//opening raised sine window file
	if (f5 != NULL)
	{
    	while (!feof(f5))
		{
       		fscanf(f5,"%lf",&w[count5]);
       		count5++;
    	}
 	}
 	else
 	printf("\nRaised_sine_window File cannot be opened");
 	fclose(f5);

	FILE *f7 = fopen("codebook.txt","r");				//opening codebook file
	if (f7 != NULL)
	{
    	while (!feof(f7))
		{
			for(int i=1; i<=32; i++)
			{
				for(int j=1; j<=12; j++)
				{
					fscanf(f7,"%lf",&cb[i][j]);
				}
			}
       		count7++;
    	}
 	}
 	else
 	printf("\nCodebook File cannot be opened");
 	fclose(f7);
	
}

void train()						//trains the system
{
	printf("********************TRAINING******************\n");
	for(int i=1;i<=15;i++)//30 names
	{
		nullify_model();

		for(int j=1;j<=30;j++)//30 utterances
 		{
			if(loop==1)
			{
				load_inertia_model();
			}
			else
			{
				load_average_model(i,loop);
			}
			
			char filename[160];
			sprintf(filename,"data/%d_%d.txt",i,j);
			FILE *f1 = fopen(filename, "r");
			printf("%s opened\n",filename);
			if (f1 != NULL)
			{
				count1=0;
    			while (!feof(f1))
				{
       				fscanf(f1,"%lf",&sn[count1]);
       				count1++;
    			}
 			}
 			else
 			printf("\nTrain Files cannot be opened");
 			fclose(f1);
		
			dcshift(sn,count1);
			normalize(sn,count1);
			//steady(sn,count1);
	
			int num_frames = 0;
			double samp[320];
			//k<count1-240
			for(int i=1,k=0;k<=(240*160) && k<count1-240;i++,k+=240)				//calculates ci's for 320 samples frame
			{
				num_frames++;
				for(int lmk = 0; lmk<320; lmk++)
					samp[lmk] = sn[lmk+k];
				//tn=start+k;
				calculate_ri(samp,r);
 				durbin(r,a);
 				cepstral(a,c);

 				for(int j=1;j<=12;j++)
				{
					f[i][j]=c[j];
					f[i][j]*=w[j];
				}
			}
			T = num_frames;
			tokhura(f);
			fun();
		}
		average(i);
		savemodel(i,loop);
	}
	loop++;
}

void trainopen()
{
	for(int i=1;i<=15;i++)
	{
		Mpi[i][1]=1.0;
	}

	for(int k=1;k<=15;k++)
	{
		char filename[160];
		sprintf(filename,"models/name%d_A%d_matrix.txt",k,4);
		FILE *f1 = fopen(filename, "r");			
		printf("%s read\n",filename);

		for(int i=1; i<=N; i++)
		{
			for(int j=1; j<=N; j++)
			{
				fscanf(f1, "%Le\t",&MA[k][i][j]);
			}
			fscanf(f1,"\n");
		}
		fclose(f1);
	}

	for(int k=1;k<=15;k++)
	{
		char filename[160];
		sprintf(filename,"models/name%d_B%d_matrix.txt",k,4);
		FILE *f1 = fopen(filename, "r");			
		printf("%s read\n",filename);

		for(int i=1; i<=N; i++)
		{
			for(int j=1; j<=M; j++)
			{
				fscanf(f1, "%Le\t",&MB[k][i][j]);
			}
			fscanf(f1,"\n");
		}
		fclose(f1);
	}
}

void trainlive(int s1)						//trains the system
{
	printf("********************LIVE_TRAINING******************\n");
	
	for(int i=1;i<=10;i++)
	{
		system("Recording_Module.exe 3 live_train.wav live_train.txt");

		count8=0;
		FILE *f6 = fopen("live_train.txt","r");
		if (f6 != NULL)
		{
			count6=0;
    		while (!feof(f6))
			{
       			fscanf(f6,"%lf",&ip[count8]);
       			count8++;
    		}
 		}
 		else
 		printf("\nInput File cannot be opened");
 		fclose(f6);

		dcshift(ip,count8);
		normalize(ip,count8);
		steady(ip,count8);

		int num_frames = 0;
		double samp[320];

		for(int i=0;i<500;i++)
		{
			for(int j=0;j<13;j++)
			{
				f[i][j]=0;
			}
		}
	
		for(int i=1,k=start; k<count6-320 && k<end; i++,k+=80)				//calculates ci's for 320 samples frame
		{
			num_frames++;
			for(int lmk = 0; lmk<320; lmk++)
				samp[lmk] = ip[lmk+k];
			//tn=start+k;
			calculate_ri(samp,r);
 			durbin(r,a);
 			cepstral(a,c);

 			for(int j=1;j<=12;j++)
			{
				f[i][j]=c[j];
				f[i][j]*=w[j];
			}
		}
		//printf("\nframes =%d\n ",num_frames);
		T=num_frames;

		tokhura(f);

		load_average_model(s1,4);

		funlive();

		savemodellive(s1);

		nullify_modellive();
	}
}

void test()								//tests the system using pre-recorded recordings
{
	printf("********************TESTING******************");
	int max_index;
	double max_v;
	for(int i=1;i<=15;i++)
	{
		printf("\nFor name %d:\n",i);
		for(int j=1;j<=30;j++)
 		{
			//load_inertia_model();
			char filename[160];
			sprintf(filename,"data/%d_%d.txt",i,j);
			FILE *f6 = fopen(filename, "r");

			if (f6 != NULL)
			{
				count6=0;
    			while (!feof(f6))
				{
       				fscanf(f6,"%lf",&ip[count6]);
       				count6++;
    			}
 			}
 			else
 			printf("\nTest Files cannot be opened");
 			fclose(f6);
		
			dcshift(ip,count6);
			normalize(ip,count6);
			//steady(ip,count6);
			int num_frames =0;
			double samp[320];
			for(int i=1,k=0;k<=(240*160) && k<count1-240;i++,k+=240)				//calculates ci's for 320 samples frame
			{
				num_frames++;
				for(int lmk = 0; lmk<320; lmk++)
					samp[lmk] = ip[lmk+k];
				//tn=start+k;
				calculate_ri(samp,r);
 				durbin(r,a);
 				cepstral(a,c);

 				for(int j=1;j<=12;j++)
				{
					f[i][j]=c[j];
					f[i][j]*=w[j];
				}
			}
			T=num_frames;
			tokhura(f);
			test_forward(i);
		}
	}

	accuracy=(count*100)/450;
	printf("\n\nAccuracy: %.2lf %%",accuracy);
}

void testlive()					//tests the system using live recordings
{
	printf("********************LIVE TESTING******************");
	int max_index,t=0;
	double max_v;

	system("Recording_Module.exe 3 input_file.wav input_file.txt");
	load_inertia_model();
	count6=0;
	FILE *f6 = fopen("input_file.txt","r");
	if (f6 != NULL)
	{
		count6=0;
    	while (!feof(f6))
		{
       		fscanf(f6,"%lf",&ip[count6]);
       		count6++;
    	}
 	}
 	else
 	printf("\nInput File cannot be opened");
 	fclose(f6);

	dcshift(ip,count6);
	normalize(ip,count6);
	steady(ip,count6);

	int num_frames = 0;
	double samp[320];
	for(int i=0;i<500;i++)
	{
		for(int j=0;j<13;j++)
		{
			f[i][j]=0;
		}
	}
	
	for(int i=1,k=start; k<count6-320 && k<end ;i++,k+=80)				//calculates ci's for 320 samples frame
	{
		num_frames++;
		for(int lmk = 0; lmk<320; lmk++)
			samp[lmk] = ip[lmk+k];
		//tn=start+k;
		calculate_ri(samp,r);
 		durbin(r,a);
 		cepstral(a,c);

 		for(int j=1;j<=12;j++)
		{
			f[i][j]=c[j];
			f[i][j]*=w[j];
		}
	}
	//printf("\nframes =%d\n ",num_frames);
	T=num_frames;
	tokhura(f);
	test_forwardlive();
}

int _tmain(int argc, _TCHAR* argv[])
{
	int s=0,s2=0;
	readfiles();
	for(int i=1;i<=3;i++)
	{
		train();
	}
	
	//trainopen();

	while(1)
	{
L1:		count=0;
		int a=1, b=0;
		
		printf("\nEnter 1 for Pre-recorded Testing\nEnter 2 for Live Testing\nEnter 3 for Live Training\n");
		scanf("%d",&s);
		switch(s)
		{
		case 1: test();break;
		case 2: 
			testlive();
			break;
		case 3:
			printf("\nEnter name as index: ");
			scanf("%d", &a);
			trainlive(a);
			break;
		default: printf("\nInvalid Choice\n");break;
		}

L3:		printf("\nEnter 1 to CONTINUE\nEnter 2 to EXIT\n");
		scanf("%d",&s2);
		switch(s2)
		{
			case 1: goto L1;break;
			case 2: goto L2;break;
			default: printf("\nInvalid Choice\n");goto L3;break;
		}
L2:		break;
	}

	//getch();
	//printmodel();
	return 0;
}
