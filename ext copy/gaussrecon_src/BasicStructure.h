#ifndef	BASIC_STRUCTURE_HEADER
#define BASIC_STRUCTURE_HEADER

#define EPSILON 1e-6
#define NON_UNIT_NORM 0
#define NEW_SAMPLES_PER_NODE 1
#define DISK_DIVISION 20

#include <limits>

struct Point{
	float x;
	float y;
	float z;
	Point(){
		x = 0; y = 0; z = 0;
	}
	Point(float xx, float yy, float zz){
		x = xx; 
		y = yy;
		z = zz;
	}
};
struct NormalPoint{
	float x;
	float y;
	float z;
	float nx;
	float ny;
	float nz;
	float area;
	float weight;
	float value;
	float width;
	NormalPoint(){
		x = y = z = nx = ny = nz = area = value = weight = width = 0;
	}
	NormalPoint(float xx, float yy, float zz, float nxx, float nyy, float nzz, float areaa, float valuee){
		x = xx; y = yy; z = zz; nx = nxx; ny = nyy; nz = nzz; area = areaa; value = valuee;
	}
	NormalPoint(float xx, float yy, float zz, float nxx, float nyy, float nzz){
		x = xx; y = yy; z = zz; nx = nxx; ny = nyy; nz = nzz;
	}
};

struct BoundingBox{
	float xscale;
	float yscale;
	float zscale;
	float blx;		// bottom-left x
	float bly;		// bottom-left y
	float blz;		// bottom-left z
};


#endif