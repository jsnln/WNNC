#ifndef ANN_ADAPTER_HEADER
#define ANN_ADAPTER_HEADER

#include <vector>
#include <ANN/ANN.h>
#include "BasicStructure.h"

using namespace std;

class ANNAdapter{
public:
	static ANNpointArray SR2ANNPointArray(vector<NormalPoint>& points, int dim = 3);
	static ANNpoint SR2ANNPoint(NormalPoint& point, int dim = 3);
	static bool deallocANNPointArray(ANNpointArray apa);
	static bool deallocANNPoint(ANNpoint ap);
};

#endif