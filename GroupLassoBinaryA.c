#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>


void MatMatProd(int m, int n, int k, int *A, double *B, double *AB);
// Matrix Matrix Product. A is in R^{mxn} and B is in R^{nxk}
void MatMatProdBinaryA(int m, int n, int k, bool *A, double *B, double *AB);
// Matrix Matrix Product. A is in R^{mxn} and B is in R^{nxk}. It is for the case where A is binary matrix 
// no floating point multiplication, only summations for this function
void getAB(bool *A, int *B,int n,int blcklngth,int M,int D,int K,int my_rank);
// To obtain A and B matrix in Renyi Classifier, 
// A is the Feature Vector data and B is the indication encoding of Labels (you can multi-class classification (more than binary))
void shrink(double lambda,double tau,int n, int m, double *Z,double *ZShrinked);
// Shrinkage operator for the matrix Z, the output is in ZShrinked
double CalcFrobNorm(int n,int m,double *Z);
// Calculating the Frobenius norm of Z

const int MAX_ITER  = 30000;
const double lambda = 30000;
// Lambda variable in Group lasso
const double tau = 5000;
const double stepsize = .9;
const double epsilon = 1e-4;


int main(int argc, char **argv)
{
    FILE *fp , *fp1;
    double frobnorm,obj,tempdouble;
    char s[200],vchar;
    int K,D,n,M,my_rank,size,i,j,D_total,blcklngth,temp,d,iter,dd;
    MPI_Status status; // for MPI receive function
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size );
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    // Obtaining the input dimensions
    // n : number of samples
    // D_total: total number of features
    // K: Number of classes/Labels
    // M: Alphabet size for feature variables
    fp  = fopen("./Data/dimensions.txt","r");
    fscanf (fp, "%d", &K);
    fscanf (fp, "%d", &D_total);
    fscanf (fp, "%d", &n);
    fscanf (fp, "%d", &M);
    fclose(fp);


    // Finding the local dimension of the feature vector A
    blcklngth = (int) ceil((D_total+0.0)/size);
    if (my_rank < size-1)
        D = blcklngth;
    else
        D = (D_total - my_rank*blcklngth);  // Each core number of variables

    // Reading the local feature vector
    bool *A;
    A = (bool*) calloc(n*M*D,sizeof(bool));
    
    // Reading the labels
    int *B;
    B = (int*) calloc(n*K,sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
        printf("Reading matrix A and B...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    // Read matrices A, B
    getAB(A, B, n,blcklngth, M,D,K,my_rank);



    double *AXsum;
    AXsum = (double*) calloc(n*K,sizeof(double));
    double *AXsum_loc;
    AXsum_loc = (double*)calloc(n*K,sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
        printf("Initializing the Matrices and initial calculation of the gradient...\n");
    MPI_Barrier(MPI_COMM_WORLD);


    for (i=0;i<n;i++)
        for (j=0;j<K;j++) 
            AXsum_loc[i+j*n] = 0;

    double *tempAXsum;
    tempAXsum = (double*) calloc(n*K,sizeof(double));



    double *X;
    X = (double*) calloc(D*M*K,sizeof(double));
    double *Xd;
    Xd = (double*) calloc(M*K,sizeof(double));
    double *Xdnew;
    Xdnew = (double*) calloc(M*K,sizeof(double));
    double *Zd;
    Zd = (double*) calloc(M*K,sizeof(double));
    bool *AdT;
    AdT = (bool*) calloc(M*n,sizeof(bool));
    for (i=0;i<M*D;i++)
        for(j = 0; j<K;j++)
            X[i+M*D*j] = 0;


    // For computing AXsum
    for(d=0;d<D;d++)
    {   
        for(i=0;i<M;i++)
            for(j=0;j<K;j++)
            {
                Xd[i+j*M] = X[i + d*M + D*M*j];  // Xd(i,j) = X(i+dM, j)
            }
        MatMatProdBinaryA(n, M, K, &A[n*M*d], Xd, tempAXsum);
        for (i=0;i<n;i++)
            for (j=0;j<K;j++) 
                AXsum_loc[i+j*n] += tempAXsum[i+j*n];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
        printf("Begin iterations...\n");
    MPI_Barrier(MPI_COMM_WORLD);


    if (my_rank == 0)
    {
        fp = fopen("./Data/Objectives.m", "w"); 
        fprintf(fp,"ObjValues = [ \n");
    }
    
    
    // The algorithm iterations starts here
    for(iter=0;iter<MAX_ITER;iter++)
    {
        if((iter > 2)&&((iter%10000)==0))
        {
            for (i=0;i<n;i++)
                for (j=0;j<K;j++) 
                    AXsum_loc[i+j*n] = 0;
            for(d=0;d<D;d++)
            {   
                for(i=0;i<M;i++)
                    for(j=0;j<K;j++)
                    {
                        Xd[i+j*M] = X[i + d*M + D*M*j];  // Xd(i,j) = X(i+dM, j)
                    }
                MatMatProdBinaryA(n, M, K, &A[n*M*d], Xd, tempAXsum);
                for (i=0;i<n;i++)
                    for (j=0;j<K;j++) 
                        AXsum_loc[i+j*n] += tempAXsum[i+j*n];
            }
        }

        MPI_Allreduce(AXsum_loc,AXsum,n*K,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 

        d = (1+iter)%D;

        if ((iter%500) == 0)
        {

            frobnorm = 0;
            for (dd=0;dd<D;dd++)
            {
                for(i=0;i<M;i++)
                    for(j=0;j<K;j++)
                        Xd[i+j*M] = X[i + dd*M + D*M*j];  // Xd(i,j) = X(i+dM, j)
                frobnorm += CalcFrobNorm(M,K,Xd);
            }
            MPI_Allreduce(&frobnorm,&obj,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

            if (my_rank == 0)
            {
                printf("Iteration Number = %d\n",iter);
                obj = lambda * obj;
                for(i=0;i<n;i++)
                    for(j=0;j<K;j++)
                        tempAXsum[i+j*n] = AXsum[i+j*n] - B [i+j*n];
                tempdouble = CalcFrobNorm(n,K,tempAXsum);
                tempdouble = tempdouble *tempdouble;
                obj += 0.5 *  tempdouble;
                fprintf(fp, "%.5f\n", obj);
            }
        }

        for(i=0;i<M;i++)
            for(j=0;j<K;j++)
                Xd[i+j*M] = X[i + d*M + D*M*j];  // Xd(i,j) = X(i+dM, j)

        for (i=0;i<n;i++)
            for(j=0;j<K;j++)
                tempAXsum[i+n*j] = AXsum[i+n*j] - B[i+n*j];
        for(i=0;i<M;i++)
            for(j=0;j<n;j++)
                AdT[i+M*j] = A[j + (d*M+i)*n];     //AdT[i,j] = Ad[j,i] = A[j,d*M+i]
        MatMatProdBinaryA(M, n, K, AdT, tempAXsum, Zd);


        for(i=0;i<M;i++)
            for(j=0;j<K;j++)
                Zd[i+M*j] = Xd[i+j*M] - 1/tau *Zd[i+M*j];
        shrink(lambda,tau,M,K,Zd,Xdnew);


        for(i=0;i<M;i++)
            for(j=0;j<K;j++)
                X[i + d*M + D*M*j] = (1-stepsize) * X[i + d*M + D*M*j] + stepsize * Xdnew[i+j*M];  //  X(i+dM, j)  = Xd(i,j) 
        
        for(i=0;i<M;i++)
            for(j=0;j<K;j++)
                Xd[i+M*j] = stepsize*(Xdnew[i+M*j] - Xd[i+M*j]);
        MatMatProdBinaryA(n, M, K, &A[n*M*d], Xd, tempAXsum);

        for(i=0;i<n;i++)
            for(j=0;j<K;j++)
                AXsum_loc[i+j*n] = AXsum_loc[i+j*n] +tempAXsum[i+j*n];
    }




    if(my_rank == 0)
    {
        fprintf(fp, "];");
        fclose(fp);
    }

    // writing the output
    mkdir("./Data/XMatrices/",0777);
    sprintf(s,"./Data/XMatrices/Indices%d",my_rank);
    fp = fopen(s,"w");
    sprintf(s,"./Data/XMatrices/XMatrix%d",my_rank);
    fp1 = fopen(s,"w");
    for(d=0;d<D;d++)
    {
        for(i=0;i<M;i++)
            for(j=0;j<K;j++)
                Xd[i+j*M] = X[i + d*M + D*M*j];  // Xd(i,j) = X(i+dM, j)
        if (CalcFrobNorm(M,K,Xd)>epsilon)
        {
            fprintf(fp, "%d\n",1 + my_rank * blcklngth + d); // 1 is for changing zero index to one
            for (i=0;i<M;i++)
            {
                for (j=0;j<K;j++)
                {
                    fprintf(fp1,"%.3f ",Xd[i+M*j]);
                }
                fprintf(fp1,"\n");
            }
        }
    }
    fclose(fp);
    fclose(fp1);

    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        fp1 = fopen("./Data/XMatrices/XMatrix.m","w");
        fprintf(fp1,"X = [\n");
        for(i=0;i<size;i++)
        {
            sprintf(s,"./Data/XMatrices/XMatrix%d",i);
            fp = fopen(s,"r");
            vchar = fgetc(fp);
            while(vchar!=EOF)
            {
                if (vchar == '\n')
                    putc(';',fp1);
                putc(vchar,fp1);
                vchar = fgetc(fp);
            }
            fclose(fp);
        }
        putc(']',fp1);
        putc(';',fp1);
        fclose(fp1);

        fp1 = fopen("./Data/XMatrices/Indices.m","w");
        fprintf(fp1,"Index = [\n");
        for(i=0;i<size;i++)
        {
            sprintf(s,"./Data/XMatrices/Indices%d",i);
            fp = fopen(s,"r");
            vchar = fgetc(fp);
            while(vchar!=EOF)
            {
                putc(vchar,fp1);
                vchar = fgetc(fp);
            }
            fclose(fp);
        }
        putc(']',fp1);
        putc(';',fp1);
        fclose(fp1);

    }   




    free(Xd);
    free(Xdnew);
    free(X);
    free(AXsum);
    free(AXsum_loc);
    free(tempAXsum);
    free(Zd);
    free(AdT);
    free(A);
    free(B);


    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;




}


void shrink(double lambda,double tau,int n, int m, double *Z,double *ZShrinked)
{
    double frobnorm,tempdouble;
    int i,j;
    frobnorm = CalcFrobNorm(n,m,Z);
    tempdouble = 1- lambda/(tau * frobnorm);
    if (tempdouble >0)
        for(i=0;i<n;i++)
            for(j=0;j<m;j++)
                ZShrinked[i+n*j] = tempdouble * Z[i+n*j];
    if ((tempdouble <=0)||(frobnorm ==0))
        for(i=0;i<n;i++)
            for(j=0;j<m;j++)
                ZShrinked[i+n*j] = 0;
}   

double CalcFrobNorm(int n,int m,double *Z)
{
    int i,j;
    double frobnorm = 0; 
    for(i=0;i<n;i++)
        for (j=0;j<m;j++)
            frobnorm += Z[i + n*j] * Z[i+n*j];
    frobnorm = sqrt(frobnorm);
    return frobnorm;
}
// integer A mxn , double B nxk
void MatMatProd(int m, int n, int k, int *A, double *B, double *AB)
{
    int i,j,ell;
    double temp;
    for (i=0;i<m;i++)
        for (j=0;j<k; j++)
        {
            temp = 0;
            for (ell = 0;ell<n;ell++)
                temp += A[i+ell*m]*B[ell+j*n];
            AB[i+m*j] = temp;
        }
}

void MatMatProdBinaryA(int m, int n, int k, bool *A, double *B, double *AB)
{
    int i,j,ell;
    double temp;
    for (i=0;i<m;i++)
        for (j=0;j<k; j++)
        {
            temp = 0;
            for (ell = 0;ell<n;ell++)
                if (A[i+ell*m])
                   temp += B[ell+j*n];
            AB[i+m*j] = temp;
        }
}


void getAB(bool *A, int *B,int n,int blcklngth,int M,int D,int K, int my_rank)
{
    FILE *fp;
    int j,i,tempint;
    char tempbool;
    fp = fopen("./Data/A.bin", "rb");
    for (j=0; j < (my_rank * n * blcklngth * M);j++)
    {
        fread((void*)(&tempbool), sizeof(tempbool), 1, fp);
    }
    for (j=0;j<D*M;j++)
    {
        for (i=0;i<n;i++)
        {
            fread((void*)(&tempbool), sizeof(tempbool), 1, fp);
            if (tempbool == 1)
                A[i+ n*j] = true;
            else
                A[i+ n*j] = false;
        }
    
    }
    fclose(fp);

    fp = fopen("./Data/B.bin", "rb");
    for (j=0;j<K;j++)
    {
        for (i=0;i<n;i++)
        {
            fread((void*)(&tempint), sizeof(tempint), 1, fp);
            B[i+ n*j] = tempint;
        }
    
    }
    fclose(fp);


}





