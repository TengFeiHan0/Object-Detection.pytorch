#include  <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>

static void sort( int n, const float* x, int* indices){
    int i, j;
    for(i=0; i<n; i++){
        for(j=i+1; j<n; j++){
            if(x[indices[j]]> x[indices[i]]){
                std::swap(indices[i], indices[j]);
            }

        }
    }

}

int nonMaximumSuppression(
    int numBoxes, const CvPoint *points,
    const CvPoint *oppositePoints, const float *score,
    float overlapThreshold, int *numBoxOut, CvPoint **pointsOut,
    CvPoint **oppositePointsOut, float *scoreOut){


        int i, j, index;
        float *box_area = (float *)malloc(numBoxes*sizeof(float));
        int *indices = (int *)malloc(numBoxes*sizeof(int));
        int *is_suppressed = (int *)malloc(numBoxes*sizeof(int));

        for(i=0; i<numBoxes; i++){
            indices[i] = i;
            is_suppressed[i] =0;
            box_area[i] = (float)( oppositePoints[i].x - points[i].x +1)*
                                  (oppositePoints[i].y - points[i].y+1);

        }

        sort(numBoxes, score, indices);

        for(int i=0; i< numBoxes; i++){
            if(!is_suppressed[indices[i]]){
                for(int j = i+1; j< numBoxes; j++){
                    if(!is_suppressed[indices[j]]){

                        int x1max = max(points[indices[i]].x, points[indices[j]].x);

                        int x2min = min(oppositePoints[indices[i]].x, oppositePoints[indices[j]].x);

                        int y1max = max(points[indices[i]].y, points[indices[j]].y);

                        int y2min = min(oppositePoints[indices[i]].y, oppositePoints[indices[j]].y);

                        int overlapWidth = x2min - x1max +1;
                        int overlapHeight = y2min - y1max +1;

                        if(overlapWidth > 0 && overlapHeight >0){
                            float overlapPart = (overlapWidth * overlapHeight)/ box_area[indices[i]] ;
                            if(overlapPart > overlapThreshold){
                                is_suppressed[indices[i]] =1;
                            }
                        }
                    }
                }
            }
        }

        *numBoxOut = 0;
        for(int i = 0; i < numBoxes; i++){
            if(!is_suppressed[i]) (*numBoxOut)++;
        }

        *pointsOut = (CvPoint*)malloc((*numBoxOut) * sizeof(CvPoint));
        *oppositePointsOut = (CvPoint*)malloc((*numBoxOut) *sizeof(CvPoint));
        *scoreOut = (float*)malloc((*numBoxOut) * sizeof(float));

        index =0;

        for(int i = 0; i < numBoxes; i++){
            if(!is_suppressed[indices[i]]){
                (*pointsOut)[index].x = points[indices[i]].x;
                (*pointsOut)[index].y = points[indices[i]].y;
                (*oppositePointsOut)[index].x = oppositePoints[indices[i]].x;
                (*oppositePointsOut)[index].y = oppositePoints[indices[i]].y;
                (*scoreOut)[index] = score[indices[i]];
                index++;
            }
        }

        free(indices);
        free(box_area);
        free(is_suppressed);
        return 1;
    }