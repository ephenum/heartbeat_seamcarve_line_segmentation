

#include "include/seam_carving.h"



using namespace cv;

using namespace std;



int main(int argc, char * argv [])

{

	//cv::namedWindow("for testing");

    if(argc <6) {

        std::cout << "Invalid number of arguments" << std::endl;

        std::cout << "Arguments are: " << std::endl;

        std::cout << "1- Entry Filename " << std::endl;

        std::cout << "2- Output Filename " << std::endl;

        std::cout << "3- Number of columns r (int)" << std::endl;

        std::cout << "4- Cubic smoothing term b (float)" << std::endl;

        std::cout << "5- Gaussian sigma smoothing term (float) " << std::endl;

        std::cout << "6- Optional Line Completion (int 0 or 1) " << std::endl;

        std::cout << "7- Optional SeamCarving Optimization size (int 1 or bigger) " << std::endl;

        std::cout << "8- Optional csv output filename " << std::endl;

		//cv::waitKey(0);

        return 0;

    }

	std::cout << string(argv[1]) << std::endl;

    cv::Mat img = cv::imread(string(argv[1]), 1);

    if(img.rows<2) {

        std::cout << "Filename does not exist" << std::endl;

		//cv::waitKey(0);

        return 0;

    }

	//Seam Carving parameters



    std::string rs(argv[3]);

    std::string bs(argv[4]);

    std::string sigmas(argv[5]);

    std::string text_file="";

    std::string::size_type sz;

    bool complete_lines = false;

    int optim =1;

    if(argc>=7) {

        std::string comp(argv[6]);

        int compi = int(std::stof(comp,&sz));

        if(compi >=1) complete_lines = true;

    }

    if(argc>=8) {

        std::string optim_size(argv[7]);

        int optim_sizei = int(std::stof(optim_size,&sz));

        if(optim_sizei >=1) optim = optim_sizei;

    }

    if(argc>=9) {

        std::string file(argv[8]);

        text_file=file;

    }

    int r = int(std::stof(rs,&sz));

    float b = std::stof(bs,&sz);

    float sigma = std::stof(sigmas,&sz);



    std::cout << "Computing seam carving with parameters: " << std::endl;

    std::cout << "r " << r << std::endl;

    std::cout << "b " << b << std::endl;

    std::cout << "sigma " << sigma << std::endl;



    Carve::SeamCarving seam(r,b,sigma,img,complete_lines, optim,text_file, true);

	seam.computeMedialSeam();

    seam.drawMedialSeam(img);

    seam.computeCarvedSeams();

    seam.drawCarvedSeam(img);

    seam.writeMatches();

    cv::imwrite(string(argv[2]), img);



    std::cout << "Seam Carving Done!!" << std::endl;

	//cv::waitKey(0);

	return 0;

}

