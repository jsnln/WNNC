#include "ANNAdapter.h"

ANNpointArray ANNAdapter::SR2ANNPointArray(vector<NormalPoint>& points, int dim){
	const int pointCount = points.size();
	ANNpointArray dataPts = annAllocPts(pointCount, dim);
	for(int i = 0; i < pointCount; i++){
		dataPts[i][0] = points[i].x;
		dataPts[i][1] = points[i].y;
		dataPts[i][2] = points[i].z;
	}
	return dataPts;
}

ANNpoint ANNAdapter::SR2ANNPoint(NormalPoint& point, int dim /* = 3 */){
	ANNpoint p = annAllocPt(dim);
	p[0] = point.x;	p[1] = point.y; p[2] = point.z;
	return p;
}

bool ANNAdapter::deallocANNPoint(ANNpoint ap){
	if(ap == NULL)
		return false;
	annDeallocPt(ap);
	return true;
}

bool ANNAdapter::deallocANNPointArray(ANNpointArray apa){
	if(apa == NULL)
		return false;
	annDeallocPts(apa);
	return true;
}