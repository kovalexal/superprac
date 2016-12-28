#ifndef ACCESSORIES_H
#define ACCESSORIES_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TRUE 1
#define FALSE 0
#define PrintStep 10
#define EPS 1e-4
#define WARPSIZE 32

#define Max(A,B) ((A)>(B)?(A):(B))
#define Min(A,B) ((A)>(B)?(B):(A))
#define R2(x,y) ((x)*(x)+(y)*(y))
#define Square(x) ((x)*(x))
#define Cube(x) ((x)*(x)*(x))

#define x(i,h0,A0) ((i)*h0 + A0)
#define y(j,h1,B0) ((j)*h1 + B0)

#define LeftPart(P,i,j,N0,N1,h0,h1)\
((-(P[N1*(i)+j+1]-P[N1*(i)+j])/h0+(P[N1*(i)+j]-P[N1*(i)+j-1])/h0)/h0+\
 (-(P[N1*(i+1)+j]-P[N1*(i)+j])/h1+(P[N1*(i)+j]-P[N1*(i-1)+j])/h1)/h1)

#define SAFE_CUDA(err) do \
{\
	if (err != 0) {\
		printf("ERROR [%s] in line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(1); \
	} \
} while (0)

//#define DecompositionPrint
#define SolutionPrint

#endif