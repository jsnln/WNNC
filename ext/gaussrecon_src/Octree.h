/*
Copyright (c) 2018, Wenjia Lu, Zuoqiang Shi, Jian Sun and Bin Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of Tsinghua University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef OCTREE_HEADER
#define OCTREE_HEADER

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <stack>
#include "ReconOctNode.h"
#include "Constants.h"
#include "Mesh.h"
#include "ANNAdapter.h"


using namespace std;
class FaceEdgesFunction;


class Octree{
public:
	int maxDepth;
	BoundingBox bb;
	static vector<NormalPoint> samplePoints;
	vector<ReconOctNode*> leafVector;

	vector<float> normalLengths;
	unordered_map< long long, GridData> gridValueMap;
	vector<GridData*> gridDataVector;
	ReconOctNode root;
	NeighborKey neighborKey;
	
	static /*const*/ int smoothIter;
	
	float minStep;
	float maxScale;
	float scaleFactor;

	int counter;
	int counterGrid;
	int counterIso;
	int gridIdx;

	vector<ReconOctNode*>::iterator leafIter;
	int threadCounter;

private:
	bool readFile(const string filename, int min_depth);	
	void estimateSampleArea2(int num_neighbors);
	float getKMaxDist2(ANNkd_tree* kdtree, ANNpoint queryPt, ANNidxArray nnidx, ANNdistArray nndists, int k );
	
	void SetIsoSurfaceCorners( const float& isovalue, const int& subdivisionDepth, const int& fullDepthIso);
	int SetMCRootPositions(ReconOctNode* node,const int& sDepth,const float& isoValue,
		unordered_map<long long,int>& boundaryRoots,unordered_map<long long,int>* interiorRoots,
		unordered_map<long long,std::pair<float,Point > >& boundaryNormalHash,unordered_map<long long,std::pair<float,Point> >* interiorNormalHash,
		std::vector<Point >* interiorPositions,
		CoredVectorMeshData* mesh,const int& nonLinearFit);
	int GetMCIsoTriangles(ReconOctNode* node, CoredVectorMeshData* mesh,unordered_map<long long,int>& boundaryRoots,
		unordered_map<long long,int>* interiorRoots,std::vector<Point>* interiorPositions,const int& offSet,const int& sDepth , bool addBarycenter , bool polygonMesh );

	float NonLinearUpdateWeightContribution(ReconOctNode* node,const float* position, const float& weight);
	void NonLinearSplatOrientedPoint(NormalPoint& position,const float& normal,const int& splatDepth,const float& samplesPerNode,const int& minDepth,const int& maxDepth, const int& currentIdx);
	void NonLinearGetSampleDepthAndWeight(ReconOctNode* node, NormalPoint* position,const float& samplesPerNode,float& depth,float& weight);
	float NonLinearGetSampleWeight(ReconOctNode* node,const float* position);
	int NonLinearSplatOrientedPoint(ReconOctNode* node,const NormalPoint& position,const float& normal);
	int HasNormals(ReconOctNode* node,const float& epsilon);
	void ClipTree();
	void setGridNode(ReconOctNode* currentNode);

	static bool isovalueComparer(const ReconOctNode* ReconOctNode1, const ReconOctNode* ReconOctNode2);
public:
	Octree();
	bool setTree(const string filename, int depth, int min_depth);
	void initLeaf();
	void GetMCIsoTriangles( const float& isovalue, CoredVectorMeshData* mesh, const int& fullDepthIso, const int& nonLinearFit, bool addBarycenter, bool polygonMesh);
	static int IsBoundaryEdge(const ReconOctNode* node,const int& dir,const int& x,const int& y,const int& subidivideDepth);
	static int IsBoundaryEdge(const ReconOctNode* node,const int& edgeIndex,const int& subdivideDepth);
	static int IsBoundaryFace(const ReconOctNode* node,const int& faceIndex,const int& subdivideDepth);

	int GetRoot(const RootInfo& ri,const float& isoValue,const int& maxDepth,Point & position,unordered_map<long long,std::pair<float,Point > >& normalHash,
		Point* normal,const int& nonLinearFit);
	int GetRoot(const RootInfo& ri,const float& isoValue,Point & position,unordered_map<long long,std::pair<float,Point> >& normalHash,const int& nonLinearFit);
	void GetMCIsoEdges( ReconOctNode* node,unordered_map<long long,int>& boundaryRoots,unordered_map<long long,int>* interiorRoots,const int& sDepth,
		std::vector<std::pair<long long,long long> >& edges );
	static int GetRootIndex(const ReconOctNode* node,const int& edgeIndex,const int& maxDepth,RootInfo& ri);
	static int GetRootIndex(const long long& key,unordered_map<long long,int>& boundaryRoots,unordered_map<long long,int>* interiorRoots,CoredPointIndex& index);
	static int GetRootPair(const RootInfo& root,const int& maxDepth,RootInfo& pair);
	static int GetEdgeLoops(std::vector<std::pair<long long,long long> >& edges,std::vector<std::vector<std::pair<long long,long long> > >& loops);
	static int AddTriangles( CoredVectorMeshData* mesh , std::vector<CoredPointIndex>& edges , std::vector<Point >* interiorPositions , const int& offSet , bool addBarycenter , bool polygonMesh );

	void writePolygon(CoredVectorMeshData* mesh, string& filename);
	void writePolygon2(CoredVectorMeshData* mesh, char* filename);

	// void loadImplicitFunctionFromNPY(std::string npyFileName, int N_grid);
	// void loadGridWidthFromNPY(std::string npyFileName, int N_grid);
};

class FaceEdgesFunction :public NodeAdjacencyFunction{
public:
	int fIndex, maxDepth;
	vector<pair<long long, long long>>* edges;
	unordered_map<long long, pair<RootInfo, int> >* vertexCount;
	void Function(const ReconOctNode* node1);
};


#endif
