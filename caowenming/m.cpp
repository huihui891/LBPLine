
#include "lbp.h"
#include "fcm.h"
#include "textureD.h"
#include <vector>

const int WSize = 24;
const int WStep = 6;

const int CLOW = 100;
const int CHIG = 250;

//����ǿ����������
const int num_points      = 1715;
const int num_dim         = 3;
const int num_cluster     = 2;
const double epsilon      = 0.05;
const double fuzziness    = 2;



int main()
{
	double start = double(getTickCount());

	//1 ����LBPͼ
	Mat img = imread("ori.jpg",0);
	Mat imgLBP = elbp(img,3,8);
	//imwrite("imgLBP.jpg",imgLBP);

	//2 ����Cannyͼ
	Mat imgCy;
	Mat RoI = img(Rect(3,3,img.cols-6,img.rows-6));
	Canny(RoI,imgCy,CLOW,CHIG);
	//imwrite("canny.jpg",imgCy);

	//���Сͼ��
	int heigh = imgLBP.rows;
	int width = imgLBP.cols;
	double area  = WSize * WSize;
	int index = 0;
	vector<vector<double>> dataBase(num_points);
	for (int i = 0; i < num_points; i++)
		dataBase[i].resize(3);
	for (int i = 0; i < heigh - WSize; i += WStep)
	{
		for (int j = 0; j < width - WSize; j += WStep)
		{
			Rect r(j,i,WSize,WSize);
			Mat subLBP = imgLBP(r);
			Mat subCy  = imgCy(r);

			//3 ����ֱ��ͼ������
			Mat hist = histc(subLBP,0,58,false);
			//printMat<float>(hist);
			double e[2] = {0.0};
			energy<float>(hist,e);

			//4 ���������ܶ�
			//printMat<uchar>(subCy);
			double ratio =  calTtureDsity(subCy,CLOW,CHIG,area);

			//5 ������������
			dataBase[index][0] = e[0];
			dataBase[index][1] = e[1];
			dataBase[index][2] = ratio;
		
			index++;
		}
	}


	//6 FCM������
 	CFCM f(num_points,num_dim,num_cluster,epsilon,fuzziness);
 	f.fcm(dataBase);
/* 	//��ӡ������ֵ
	char *of = "lishudu.txt";
	ofstream fout(of, ios::out | ios::app);
	if (!fout.is_open())
	{
		cerr << "Can't open " << of << "file for output!\n";
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < num_points; i++)
	{
		for (int j = 0; j < num_cluster; j++)
		{
			fout << f.degree_of_memb[i][j] << " ";
		}
		fout << endl;
	}
	*/


	//7 ������������ֵ ��0.5Ϊ��׼
	Mat segImg(imgLBP.size(),CV_8UC1);
	segImg.setTo(0);
	for (int i = 0; i < num_points; i++)
	{
		if (f.degree_of_memb[i][0] > 0.5)
		{
			//�����ǵڶ���
			//��������ͼ���е�λ��
			int m = i / 48 * WStep;
			int n = i % 48 * WStep;
			for (int p = m; p < m + WSize; p++)
			{
				uchar *k = segImg.ptr<uchar>(p);
				for (int q = n; q < n + WSize; q++)
					k[q] = 255;
			}
		}
	}

	double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
	cout << "It took " << duration_ms << " ms." << endl;
	imshow("seg",segImg);

	cvWaitKey(0);

	return 0;
}

