#ifndef	CUBE_HEADER
#define CUBE_HEADER

#include "BasicStructure.h"

class Cube{
public:
	enum NEIGHBOR_DIRE{X_FRONT, X_BACK, Y_FRONT, Y_BACK, Z_FRONT, Z_BACK};
	const static int CORNERS = 8;
	const static int EDGES = 12;
	const static int NEIGHBORS = 6;
	const static int CornerAdjacentMap[CORNERS][NEIGHBORS];

	static int CornerIndex( const int& x, const int& y, const int& z);
	static int CornerIndex( const float* center, const float* position);
	static int CornerIndex( const float* center, const NormalPoint* np);
	static void FactorCornerIndex( const int& idx,int& x,int& y,int& z);
	static void FactorFaceIndex		(const int& idx,int& x,int &y,int& z);
	static void FactorFaceIndex		(const int& idx,int& dir,int& offSet);
	static int  AntipodalCornerIndex	(const int& idx);
	static void EdgeCorners(const int& idx,int& c1,int &c2);
	static void FactorEdgeIndex		(const int& idx,int& orientation,int& i,int &j);
	static int  EdgeIndex			(const int& orientation,const int& i,const int& j);
	static void FacesAdjacentToEdge	(const int& eIndex,int& f1Index,int& f2Index);
	static int  FaceIndex			(const int& x,const int& y,const int& z);
	static int  FaceReflectEdgeIndex	(const int& idx,const int& faceIndex);
	static int	EdgeReflectEdgeIndex	(const int& edgeIndex);
	static int	FaceReflectFaceIndex	(const int& idx,const int& faceIndex);
	static int  FaceAdjacentToEdges	(const int& eIndex1,const int& eIndex2);
	static void FaceCorners(const int& idx,int& c1,int &c2,int& c3,int& c4);

};
#endif