//	The heartbeat-seamcarving programme serves to segment lines of a given text block by means of the heartbeat-seamcarve algorithm as published by Mathias Seuret, Daniel Stökl Ben Ezra, Marcus Liwicki. Robust Heartbeat-based Line Segmentation Methods for Regular Texts and Paratextual Elements. HIP 2017 - Proceedings of the 4th International Workshop on Historical Document Imaging and Processing, Nov 2017, Kyoto, Japan. 〈hal-01677054〉. https://hal.archives-ouvertes.fr/hal-01677054

//    Copyright (C) 2017  Mathias Seuret, Daniel Stökl Ben Ezra, Marcus Liwicki. EPHE, PSL. U Fribourg.
//    Contact at: daniel.stoekl@ephe.psl.eu, mathias.seuret@unifr.ch, marcus.liwicki@unifr.ch.

//    This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License Version 3 as published by the Free Software Foundation.

//    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

//    You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <https://www.gnu.org/licenses/lgpl-3.0.en.html>.


#ifndef __H__SEAM__CARVING__
#define __H__SEAM__CARVING__

#include <SPLINTER/datatable.h>
#include <SPLINTER/bsplinebuilder.h>
//#include "../splinter-master/include/bspline.h"
//#include "../splinter-master/include/bsplinebasis.h"
//#include <opencv/highgui.h> //opencv 3.1
//#include <opencv2/highgui.hpp> //opencv 3.1
#include <opencv2/highgui/highgui.hpp> 
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <list>
#include <fstream>

using namespace std;

/*-----------------------------------------------------------------------------*/
class TextProcessor
{
public:
	TextProcessor() { empty = true; };
	TextProcessor(TextProcessor const& transc)
	{
        lines_number = transc.lines_number;
		text_lines = transc.text_lines;
        full_text_lines = transc.full_text_lines;
		empty = transc.empty;
	};
	TextProcessor(std::string text_file) {
		std::ifstream file(text_file);
		std::ofstream fileo("test.txt");
		std::string str;
		//skipping first line
		std::getline(file, str);
		int line_no = 0;
		int end_line = 0;
		while (std::getline(file, str))
		{
            if (str.length()<2) {
				cout << "skip empty line " << endl;
				continue;
			}
            string stri = str.substr(0, std::min(21,int(str.length()-1)));//eliminate first character
			if (str.substr(0, 10) == "<pb xml:id") {
				lines_number.push_back(line_no - end_line);
				cout << "line number " << line_no - end_line << endl;
				end_line = line_no+1;
			}
			else {
				text_lines.push_back(stri);
				full_text_lines.push_back(str);
				fileo << str << endl;
			}
			line_no++;
		}
		fileo.close();
		empty = false;
		file.close();
	}
	bool empty;
	std::vector<std::string> text_lines;
	std::vector<int> lines_number;
	std::vector<std::string> full_text_lines;
};

namespace Carve {

	/*-----------------------------------------------------------------------------*/

	using namespace SPLINTER;



    const float MaximaThreshold    = 0.5f;
    const float MaximaThresholdLow = 0.25f;
    const float SpaceLowThreshold  = 0.75f;
    const float SpaceHighThreshold = 1.25f;
    const float SpaceTwoLinesThreshold = 1.7f;
    const signed int MinDistanceLines =5;
	const int MaxInsertion = 500;



	struct maximum {
        std::vector<std::vector<double> > maxima;
		std::vector<int> maxima_size;
	};



	class SeamCarving {

	public:
        SeamCarving(int r, float b, float sigma, const cv::Mat& img,bool complete_lines, int optim_size, std::string text_file, bool hb);
        //SeamCarving() {};
		~SeamCarving() ;
		void computeMedialSeam();
		void applyWaveLength(std::vector<std::vector<cv::Point> >& medial);
		void computeCarvedSeams();
		void drawMedialSeam(cv::Mat& img);
		void drawCarvedSeam(cv::Mat& img);
        void writeMatches();

	private:
        void								computeProjection(std::vector<std::vector<int> >& SobelProjection, const cv::Mat& absImgSobel);
        maximum								computeSpline(std::vector<std::vector<int> >& SobelProjection, std::vector<DataTable>& splines) ;
        std::vector<std::vector<cv::Point> > computeSymmetricMatch(const maximum& optimum);
		cv::Point							computeMatch(cv::Point p_start,const maximum& p_list,int slice);
        void								printMatches(const std::vector<std::vector<cv::Point> >& matches);
        void								convertToMedialSeam(const std::vector<std::vector<cv::Point> >& matches);
		int									getRowCoordonate(cv::Point p_start, cv::Point p_end, int col);
		void								fillMedialSeam(cv::Point p_start, cv::Point p_end,int line);
		void								computeEnergyMap(const cv::Mat& imgSmooth);
        void								computeForwardEnergy(std::vector<std::vector<cv::Point> >& Energy,int min_index,int max_index,int line);
        void								computeBackwardEnergy(std::vector<std::vector<cv::Point> >& Energy, int min_index, int max_index, int line);
        float                               computeMedianSpace(std::vector<std::vector<cv::Point> >& matches);
        float                               computeMeanMaxima(std::vector<std::vector<cv::Point> >& matches,std::vector<std::vector<int> >& SobelProjection);
        bool                                addMatch(std::vector<std::vector<cv::Point> >& matches,std::vector<std::vector<int> >& SobelProjection,float space, float mean_maxima);
        bool                                delMatch(std::vector<std::vector<cv::Point> >& matches,std::vector<std::vector<int> >& SobelProjection,float space, float mean_maxima);
        void                                insertMatch(std::vector<std::vector<cv::Point> >& matches,int max_r,int max_li,int l);
        void                                deleteMatch(std::vector<std::vector<cv::Point> >& matches, int l);
											// computes the mean vertical distance between two sequences of points - ms
        int 								meanDist(const std::vector<cv::Point> &a, const std::vector<cv::Point> &b);
        int 								meanDist(const std::vector<cv::Point> &a, const int b);
											// adds a number of lines from the first sequence (incl) to the second sequence (non-incl) of points - ms
        void								interpolatedInsertion(std::vector<std::vector<cv::Point> >& lines, const std::vector<cv::Point>& from, const std::vector<cv::Point>& to, int count);
        void								interpolatedInsertion(std::vector<std::vector<cv::Point> >& lines, const std::vector<cv::Point>& from, const int to, int count);
		int		_r;
		int		_w;
        int     _optim_size;
		float	_sigma;
		float	_b;
        bool    _complete_lines;
		cv::Mat _img;
		cv::Mat _imgEnergy;
        std::string                    _text_file;
        std::vector<std::vector<int> > _medial_seam;
        std::vector<std::vector<int> > _carve_seam;
        std::vector<std::vector<cv::Point> > wl;
        bool _heartbeat;
	};

	/*-----------------------------------------------------------------------------*/

    SeamCarving::SeamCarving(int r, float b, float sigma, const cv::Mat& img, bool complete_lines=false, int optim_size=1,std::string text_file ="", bool hb=true) :
		_r(r),
		_sigma(sigma),
        _b(b),
        _complete_lines(complete_lines),
        _optim_size(optim_size),
        _text_file(text_file),
        _heartbeat(hb)
	{

		if (_r < 2) _r = 2;

		if (img.channels() == 3)
			cv::cvtColor(img, _img, CV_BGR2GRAY);
		else
			img.copyTo(_img);

        if(_r >_img.cols)   _r = int(_img.cols / 2);

		// smoothing term
        _b = float(_img.rows)*float(_img.rows)*b;

        _w = int(std::floor(float(_img.cols) / float(_r)));
	}

	/*-----------------------------------------------------------------------------*/

	SeamCarving::~SeamCarving()

	{
		_img.release();
		_imgEnergy.release();
	}

	/*-----------------------------------------------------------------------------*/

    void SeamCarving::writeMatches()
    {
        if(_text_file !="" ) {
            ofstream myfile;
            myfile.open (_text_file);
            for(int i=0;i<_carve_seam.size();i++) {
                for(int j=0; j<_carve_seam[i].size();j++) {
                    myfile << _carve_seam[i][j];
                    if(_carve_seam[i].size()>1)
                        if(j<_carve_seam[i].size()-1)
                            myfile << ",";
                }
                myfile << "\n";
            }
            myfile.close();
        }
    }

    /*-----------------------------------------------------------------------------*/

/*
 * Computes projections profiles ; the vector contains one vector for each row ; the
 * row vectors contain one value per vertical slice of the image (_r).
 * */
    void SeamCarving::computeProjection(std::vector<std::vector<int> >& SobelProjection, const cv::Mat& absImgSobel)
	{
		//compute projection

		SobelProjection.resize(_img.rows);
		for (int i = 0; i < _img.rows; i++) {
			SobelProjection[i].resize(_r);
			const unsigned char* sobelRow = absImgSobel.ptr<unsigned char>(i);

			for (int r = 0; r < _r; r++) {
				int projir = 0;
				for (int k = r*_w; k < (r + 1)*_w; k++) {
					projir += int(sobelRow[k]);
				}
				SobelProjection[i][r] = projir;
			}
		}
	}

	/*-----------------------------------------------------------------------------*/

/*
 * Computes a spline for each slice of the image. Then looks for maxima.
 * */
    maximum SeamCarving::computeSpline(std::vector<std::vector<int> >& SobelProjection, std::vector<DataTable>& splines)

	{

		maximum optimum;

		optimum.maxima.resize(_r);

		optimum.maxima_size.resize(_r);

		//compute cubic spline

		for (int r = 0; r < _r; r++) {

			optimum.maxima[r].resize(_img.rows);

			double y;

			for (int i = 0; i < _img.rows; i++) {

				y = SobelProjection[i][r];

				splines[r].addSample(i, y);

			}

			BSpline pspline = BSpline::Builder(splines[r])

				.degree(3)

                .smoothing(BSpline::Smoothing::PSPLINE)

				.alpha(_b)

				.build();

			//compute local maxima

			int size_maxima = 0;

			for (int i = 1; i < _img.rows-1; i++)

			{

				DenseVector ev_prev(1);

				DenseVector ev_cur(1);

				DenseVector ev_next(1);

				ev_prev(0) = double(i-1);

				ev_cur(0) = double(i );

				ev_next(0) = double(i + 1);

				float val_prev = pspline.eval(ev_prev);

				float val_cur = pspline.eval(ev_cur);

				float val_next = pspline.eval(ev_next);



				if ((val_cur > val_prev) && (val_cur > val_next)) //get one local maxima
				{
					optimum.maxima[r][size_maxima] = i;
                    size_maxima++;
				}
			}

			optimum.maxima_size[r] = size_maxima;

		}

		return optimum;

	}

	/*-----------------------------------------------------------------------------*/

/*
 * Computes for a point p_start the closest point in a list p_list. The result point is
 * put in the middle of the given slice.
 * */
	cv::Point SeamCarving::computeMatch(cv::Point p_start, const maximum& p_list, int slice)

	{

		float dist_min = FLT_MAX;

		int index_min = 0;

		for (int lo = 0; lo < p_list.maxima_size[slice]; lo++) {

			cv::Point p_cur(0.5*(2 * slice + 1)*_w, p_list.maxima[slice][lo]);

			float dist = cv::norm(p_start - p_cur);

			if (dist < dist_min) {

				dist_min = dist;

				index_min = lo;

			}

		}

		return cv::Point(0.5*(2 * slice + 1)*_w, p_list.maxima[slice][index_min]);

	}

	/*-----------------------------------------------------------------------------*/

    std::vector<std::vector<cv::Point> > SeamCarving::computeSymmetricMatch(const maximum& optimum)

	{

        std::vector<std::vector<cv::Point> > matches;

		//process all lines

		for (int l = 0; l < optimum.maxima_size[0]; l++) {

			//process possible line l

			std::vector<cv::Point> line_points;

			cv::Point p_start(0.5*_w,optimum.maxima[0][l]);

			line_points.push_back(p_start);

			//symetric check for the whole line

			bool valid_line = true;			

			for (int r = 1; r < _r; r++) {

				//get match in slice r with p_start

				cv::Point p_match = computeMatch(p_start, optimum, r);

				cv::Point p_start_sym = computeMatch(p_match, optimum, r-1);

				if (norm(p_start - p_start_sym) > FLT_EPSILON) {

					valid_line = false;

					break;

				}

				//update p_start

				p_start = p_match;

				line_points.push_back(p_match);				

			}

			if (valid_line) matches.push_back(line_points);

		}

		return matches;

	}

	/*-----------------------------------------------------------------------------*/

    void SeamCarving::printMatches(const std::vector<std::vector<cv::Point> >& matches)

	{

		for (int l = 0; l < matches.size(); l++) {

			for (int c = 0; c < matches[l].size(); c++) {

				cout << " x: " << matches[l][c].y << " y: " << matches[l][c].x;

			}

			cout << endl;

		}

	}

	/*-----------------------------------------------------------------------------*/

	/**
	 * turns a set of a few points into a line with one point per pixel (with iterpolation)
	 * */
    void SeamCarving::convertToMedialSeam(const std::vector<std::vector<cv::Point> >& matches)

	{

		_medial_seam.resize(matches.size());

		//go through lines

		for (int l = 0; l < matches.size(); l++) {

			_medial_seam[l].resize(_img.cols);

			//first segment 

			cv::Point p_start(0,matches[l][0].y);

			cv::Point p_end(matches[l][0].x, matches[l][0].y);

			fillMedialSeam(p_start,p_end,l);

			for (int r = 0; r < _r - 1; r++) {

				cv::Point p_startg(matches[l][r].x, matches[l][r].y);

				cv::Point p_endg(matches[l][r+1].x, matches[l][r+1].y);

				fillMedialSeam(p_startg, p_endg,l);

			}

			//last segment

			cv::Point p_starte(matches[l][_r-1].x, matches[l][_r-1].y);

			cv::Point p_ende(_img.cols-1, matches[l][_r-1].y);

			fillMedialSeam(p_starte, p_ende,l);

		}

	}

	/*-----------------------------------------------------------------------------*/
/**
 * interpolates Y coordinate for all points between start and end
 * */
	void SeamCarving::fillMedialSeam(cv::Point p_start, cv::Point p_end,int line)

	{

		for (int c = p_start.x; c <= p_end.x; c++) {

			float l = getRowCoordonate(p_start, p_end, c);

			_medial_seam[line][c] = l;

		}

	}

	/*-----------------------------------------------------------------------------*/

	int SeamCarving::getRowCoordonate(cv::Point p_start, cv::Point p_end, int col)

	{

		//compute line coefficient: l= a*c +b

		float c1 = p_start.x;

		float l1 = p_start.y;

		float c2 = p_end.x;

		float l2 = p_end.y;

		float a = (l1 - l2) / (c1 - c2);

		float b = (l2*c1 - l1*c2) / (c1 - c2);

		return ( int(a * float(col) + b + 0.5));

	}

	/*-----------------------------------------------------------------------------*/

	void SeamCarving::drawMedialSeam(cv::Mat& img)

	{ 

		for (int l = 0; l < _medial_seam.size(); l++)

            for (int c = 0; c < _medial_seam[l].size()-1; c++)

				if (_medial_seam[l].size() <= img.cols)

					if (_medial_seam[l][c] >= 0)

						if (_medial_seam[l][c] < img.rows)

                            cv::line(img,cv::Point(c, _medial_seam[l][c]),cv::Point(c, _medial_seam[l][c+1]),cv::Scalar(255, 0, 0));

                            //cv::circle(img, cv::Point(c, _medial_seam[l][c]), 1, cv::Scalar(255, 0, 0), -1);

	}

	/*-----------------------------------------------------------------------------*/

	void SeamCarving::drawCarvedSeam(cv::Mat& img)

	{

		for (int l = 0; l < _carve_seam.size(); l++) {

            for (int c = 0; c <  _carve_seam[l].size()-1; c++) {

				if (_carve_seam[l].size() <= img.cols) {
					for (int y=_carve_seam[l][c]-1; y<_carve_seam[l][c]+1; y++) {
						if (y >= 0) {
							if (y < img.rows) {
								cv::line(img,cv::Point(c, y),cv::Point(c, _carve_seam[l][c+1]),cv::Scalar(0, 0,255));
							}
						}
					}
				}
			}
		}

                            //cv::circle(img, cv::Point(c, _carve_seam[l][c]), 1, cv::Scalar(0, 0, 255), -1);

	}

    /*-----------------------------------------------------------------------------*/

    float SeamCarving::computeMedianSpace(std::vector<std::vector<cv::Point> >& matches)

    {

        if(matches.size()<2) return 1.0f;

        int row    = int(std::floor(float(_r)/2.0f));

        int middle = int(std::floor(float(matches.size())/2.0f));

        std::vector<int> spaces;

        for(int l=0;l<matches.size()-1;l++) {

            int space = matches[l+1][row].y - matches[l][row].y;

            spaces.push_back(space);

        }

        std::sort(spaces.begin(),spaces.end());

        return float(spaces[middle]);

    }

	/*-----------------------------------------------------------------------------*/

    float SeamCarving::computeMeanMaxima(std::vector<std::vector<cv::Point> >& matches,std::vector<std::vector<int> >& SobelProjection)

    {

        if(matches.size() < 1)  return FLT_MAX;

        int row    = int(std::floor(float(_r)/2.0f));

        int maxima = 0;

        for(int l=0;l<matches.size();l++) {

            int y = matches[l][row].y;

            maxima += SobelProjection[y][row];

        }

        return( float(maxima) / float(matches.size()));

    }

    /*-----------------------------------------------------------------------------*/

    void SeamCarving::insertMatch(std::vector<std::vector<cv::Point> >& matches,int max_r,int max_li,int l)

    {

        std::vector<cv::Point> match;

        if(l==-1) {

            for(int r =0;r<_r;r++) {

                match.push_back(cv::Point(0.5*(2 * r + 1)*_w,max_li));

            }

        }

        else {

            int space = max_li - matches[l][max_r].y;

            for(int r =0;r<_r;r++) {

                match.push_back(cv::Point(0.5*(2 * r + 1)*_w,std::min(matches[l][r].y+space,_img.rows-1)));

            }

        }

        if(l==matches.size()-1) {

            matches.push_back(match);

            return;

        }

        std::vector<std::vector<cv::Point> > matches_new;

        for(int ln=0;ln<=l;ln++)

            matches_new.push_back(matches[ln]);

        matches_new.push_back(match);

        for(int ln=l+1;ln<matches.size();ln++)

            matches_new.push_back(matches[ln]);

        matches = matches_new;

    }

    /*-----------------------------------------------------------------------------*/

    void SeamCarving::deleteMatch(std::vector<std::vector<cv::Point> >& matches, int l)

    {

        std::vector<std::vector<cv::Point> > matches_new;

        for(int ln=0;ln<l;ln++)

            matches_new.push_back(matches[ln]);

        for(int ln=l+1;ln<matches.size();ln++)

            matches_new.push_back(matches[ln]);

        matches = matches_new;

    }

    /*-----------------------------------------------------------------------------*/

    bool SeamCarving::delMatch(std::vector<std::vector<cv::Point> >& matches,std::vector<std::vector<int> >& SobelProjection,float space, float mean_maxima)

    {

        for(int l=0;l< int(matches.size());l++) {

            //get max val

            int max_val =0;

            for(int r=0;r<_r;r++) {

                int li = matches[l][r].y;

                int val = SobelProjection[li][r];

                if(val > max_val)

                    max_val = val;

            }

            if(max_val < mean_maxima * MaximaThresholdLow) {

                //del line l

                deleteMatch(matches,l);

                return true;

            }

        }

        return false;

    }

    /*-----------------------------------------------------------------------------*/

    bool SeamCarving::addMatch(std::vector<std::vector<cv::Point> >& matches,std::vector<std::vector<int> >& SobelProjection,float space, float mean_maxima)

    {

        if(matches.size() < 1)  return false;

        bool match_added=false;

		space = std::max(1.0f,space);//if space =0 infinite loop

        int col    = int(std::floor(float(_r)/2.0f));

        for(signed int l=-1;l< int(matches.size());l++) {

            int y =0;

            if(l>-1) y= matches[l][col].y;

            signed int spacer;

            if(l==-1)                      spacer = matches[l+1][col].y;

            else if(l==matches.size()-1)   spacer = _img.rows-1 - y;

            else                           spacer = matches[l+1][col].y - y;

                int loop = 0;

                do {

                    //Try adding new line

                    //Get Maximal Projection between space

                    int start = int(float(y)+float(loop)*float(space)+ SpaceLowThreshold *float(space));

					start = std::max(start, y+1);

                    int end   = int(float(y)+float(loop)*float(space)+ SpaceHighThreshold*float(space));

                    if((float(spacer) > (SpaceTwoLinesThreshold*space +float(loop)*float(space)))&&(start < _img.rows)) {

                        int max_val=0;

                        int max_r  =0;

                        int max_li =0;

                        for(int li=start; li <= end;li++) {

                            for(int r=0; r<_r;r++) {

                                int val=0;

                                if(li < SobelProjection.size())

                                    val = SobelProjection[li][r];

                                if(val > max_val) {

                                    max_val = val;

                                    max_r   = r;

                                    max_li  = li;

                                }

                            }

                        }

                        if(max_val > mean_maxima * MaximaThreshold ) {

							match_added = true;

							//insert match: col max_r, line max_li, inserted after index l and before l+1

							insertMatch(matches,col,max_li,l);

							break;

                        }

                    }

                    loop++;

                } while ((!match_added) && (float(spacer) > (SpaceTwoLinesThreshold*space +float(loop)*space)));



            if(match_added) break;

        }

        return match_added;

    }

    /*-----------------------------------------------------------------------------*/

	/**
	 * Computes the "blue line", i.e., a polyline on the textline
	 * */
	void SeamCarving::computeMedialSeam()

	{

		// impossible, set to >=2 in the constructor
        if(_r<2) return;

		cv::Mat imgSobel;

		cv::Mat absImgSobel;

        cv::Sobel(_img, imgSobel,CV_16S,1,0);

        cv::convertScaleAbs(imgSobel, absImgSobel);

        std::vector<std::vector<int> > SobelProjection;

		computeProjection(SobelProjection,absImgSobel);

		std::vector<DataTable> splines;

		splines.resize(_r);

		maximum optimum = computeSpline(SobelProjection, splines);

        std::vector<std::vector<cv::Point> > matches;
        
        for (int m=0; m<optimum.maxima.size(); m++) {
            for (int c=0; c<optimum.maxima[m].size(); c++) {
                if (optimum.maxima[m][c]==0) {
                    continue;
                }
                cout << "Medial point: [" << m << "]: " <<optimum.maxima[m][c] << endl;
            }
        }

		matches = computeSymmetricMatch(optimum);
        
        for (int l=0; l<matches.size(); l++) {
            for (int c=0; c<matches[l].size(); c++) {
                cout << "Match: " << l << " " << matches[l][c].x << " " << matches[l][c].y << endl;
            }
        }

        if(_complete_lines) {

            float space = computeMedianSpace(matches);

            float mean_maxima = computeMeanMaxima(matches,SobelProjection);



            std::cout << "Mean Distance between Lines: " << space       << std::endl;

            std::cout << "Mean Maximal Projection    : " << mean_maxima << std::endl;

			int num_insertion = 0;

            bool match_added;

            do {

                 match_added = addMatch(matches,SobelProjection, space, mean_maxima);

            } while(match_added);

            //delete lines with small peak

            bool match_del;

            do {

                 match_del = delMatch(matches,SobelProjection, space, mean_maxima);

				 if (match_added)	num_insertion++;

            } while(match_del && (num_insertion<MaxInsertion));



        }
        
		// wavelength call - ms
		if (_heartbeat) {
			cout << "Lines detected: " << matches.size() << endl;
            // this method will fill wl with matches + additional medial lines - ms
			applyWaveLength(matches);
			cout << "Lines retrieved: " << wl.size() << endl;
			// fill the medial lines, i.e., 1 point per pixel - ms
			convertToMedialSeam(wl);
		} else {
            // no heartbeat
            // fill the medial lines, i.e., 1 point per pixel - ms
			convertToMedialSeam(matches);
			cout << "Lines retrieved: " << matches.size() << endl;
		}
		
		cout << "Step done" << endl;

	}
	
	/**
	 * Wavelength implementation - interpolating new medial lines
	 * */
	void SeamCarving::applyWaveLength(std::vector<std::vector<cv::Point> >& medial) {
		// puts all vertical peak distances into a vector - ms
		vector<int> distances;
		for (int l=0; l<medial.size()-1; l++) {
			for (int c=0; c<medial[l].size(); c++) {
				distances.push_back(medial[l+1][c].y - medial[l][c].y);
			}
		}
		cout << distances.size() << " distances found" << endl;
		
        // Sort them in an extremely inefficient way
		//TODO: replace with a quick sort - I have no offline documentation - ms
		for (int i=0; i<distances.size(); i++) {
			for (int j=i+1; j<distances.size(); j++) {
				if (distances[i]>distances[j]) {
					int k = distances[j];
					distances[j] = distances[i];
					distances[i] = k;
				}
			}
		}
		
        // Extracts the median value of the vector - ms
		int median = distances[distances.size()/2]-1;
		cout << "Median distance: " << median << endl;
		
		// For each medial line passed as parameter, look for the distance
        // to the next one and decides whether there would be space for
        // one in the middle (at least 1.5x the median distance). - ms
		for (int l=0; l<medial.size()-1; l++) {
			// md=mean distance between the medial lines at each slice - ms
            int md = meanDist(medial[l], medial[l+1]);;
			
            // estimates how many could be inserted - ms
			int insert_count = round(md/(float)median);
            
            // In case of insertion of more than the next medial line, let the
            // user know about it - ms
			if (insert_count>1) {
				cout << (insert_count-1) << " missing line(s) between L" << (l+1) << " and L" << (l+2) << " (space: " << md << "px)" << endl;
			}
            
            // interpolates the insertion; note that if insert_count==1, then
            // it inserts only the next medial line
			interpolatedInsertion(wl, medial[l], medial[l+1], insert_count);
		}
        // as the above loop processed only insertions when there is
        // a next medial line, we have to add the last medial line. - ms
		wl.push_back(medial[medial.size()-1]);
	}
	
    // returns the mean vertical distance between two vectors of points - ms
	int SeamCarving::meanDist(const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
		int sum = 0;
		for (int i=0; i<a.size(); i++) {
			sum += b[i].y - a[i].y;
		}
		return sum / a.size();
	}
	
    // returns the mean vertical distance between a vector of point and a y coordinate - ms
	int SeamCarving::meanDist(const std::vector<cv::Point> &a, const int b) {
		int sum = 0;
		for (int i=0; i<a.size(); i++) {
			sum += b - a[i].y;
		}
		return sum / a.size();
	}
	
    // inserts "count" medial lines into "lines", with interpolation between the line "from" and the line "to".
	void SeamCarving::interpolatedInsertion(std::vector<std::vector<cv::Point> >& lines, const std::vector<cv::Point>& from, const std::vector<cv::Point>& to, int count) {
		int l = lines.size();
		lines.resize(lines.size()+count);
		for (int n=0; n<count; n++) {
			lines[l+n].resize(from.size());
			for (int c=0; c<from.size(); c++) {
				int dy = round((to[c].y - from[c].y) / (float)count * n);
				lines[l+n][c].x = from[c].x;
				lines[l+n][c].y = from[c].y + dy;
			}
		}
	}
	
    // inserts "count" medial lines into "lines", with interpolation between the line "from" and the vertical coordinate "to".
	void SeamCarving::interpolatedInsertion(std::vector<std::vector<cv::Point> >& lines, const std::vector<cv::Point>& from, const int to, int count) {
		int l = lines.size();
		lines.resize(lines.size()+count);
		for (int n=0; n<count; n++) {
			lines[l+n].resize(from.size());
			for (int c=0; c<from.size(); c++) {
				int dy = round((to - from[c].y) / (float)count * n);
				lines[l+n][c].x = from[c].x;
				lines[l+n][c].y = from[c].y + dy;
			}
		}
	}

	/*-----------------------------------------------------------------------------*/

	void SeamCarving::computeEnergyMap(const cv::Mat& imgSmooth)

	{

		_imgEnergy = cv::Mat::zeros(cv::Size(_img.cols, _img.rows), CV_8U);

		for (int r = 1; r < _img.rows-1; r++) {

			const unsigned char* Rowm = imgSmooth.ptr<unsigned char>(r-1);

			const unsigned char* Row = imgSmooth.ptr<unsigned char>(r);

			const unsigned char* Rowp = imgSmooth.ptr<unsigned char>(r+1);

			unsigned char* RowE = _imgEnergy.ptr<unsigned char>(r);

			for (int c = 1; c < _img.cols - 1; c++) 

				RowE[c] = uchar (std::abs(float(Rowm[c] - Rowp[c])) + std::abs(float(Row[c - 1] - Row[c + 1])));

		}

	}

	/*-----------------------------------------------------------------------------*/

	void SeamCarving::computeCarvedSeams()

	{

		cv::Mat imgSmoothed;

		cv::GaussianBlur(_img, imgSmoothed, cv::Size(3,3), _sigma, _sigma);

		computeEnergyMap(imgSmoothed);

		//cv::imshow("Energy" , _imgEnergy);

		// not possible, set to >=2 in the constructor - ms
        if (_r < 2)                             return;

		// i.e., no line was detected - ms
		if (_medial_seam.size() <= 1)			return;

		_carve_seam.resize(_medial_seam.size()-1);

		for (int l = 0; l < _carve_seam.size(); l++)

			_carve_seam[l].resize(_img.cols);



		//compute line seam

		for (int l = 0; l < _carve_seam.size() ; l++)
		{

			//Allocate size for propagation

			int min_index = INT_MAX;

			int max_index = 0;

			for (int c = 0; c < _img.cols; c++) {

				int mini = _medial_seam[l][c];

				int maxi = _medial_seam[l+1][c];				

				if (maxi > max_index)	max_index = maxi;

				if (mini < min_index)	min_index = mini;

			}
			
			/*
			 * The min/max indices indicate the extremum points of the two
			 * medial lines around where we've to carve. - ms
			 * */

            std::vector<std::vector<cv::Point> > Energy;

			computeForwardEnergy(Energy, min_index, max_index, l);

			//Backward dynamic programming

			computeBackwardEnergy(Energy, min_index, max_index, l);

		}





	}

	/*-----------------------------------------------------------------------------*/

    void SeamCarving::computeForwardEnergy(std::vector<std::vector<cv::Point> >& Energy, int min_index, int max_index, int line)

	{

        std::vector<int> Energy_table;

        Energy_table.resize(2*_optim_size+1);



		Energy.resize(_img.cols);

		int size_energy = max_index - min_index + 1;

		for (int i = 0; i < _img.cols; i++) {

			Energy[i].resize(size_energy);

		}

		//Forward dynamic programming: Compute Min Energy

		int c = 0;

		// lp corresponds to the row index - ms
		for (int lp = 0; lp < size_energy; lp++)

		{

			unsigned char* RowE = _imgEnergy.ptr<unsigned char>(min_index + lp);

			Energy[c][lp].x = RowE[c];

		}

		for (c = 1; c < _img.cols; c++) {

			int mini = _medial_seam[line][c-1];

			int maxi = _medial_seam[line + 1][c-1];

			for (int lp = 0; lp < size_energy; lp++)

			{

                for(int idx=0;idx<2*_optim_size+1;idx++)

                    Energy_table[idx] = INT_MAX;

				unsigned char* RowE = _imgEnergy.ptr<unsigned char>(min_index + lp);

                int energy_min = INT_MAX;

                int idxr_min   = 0;

				//energy from previous column

                for(int idx=0;idx<2*_optim_size+1;idx++) {
					//TODO : multiply cost by abs(idxr)+1
                    signed int idxr = idx - _optim_size;
                    int cost = abs(idxr)+1;

                    if ((lp + idxr + min_index >= mini) && (lp + idxr + min_index <= maxi) && (lp+idxr>=0))	{

                        if(idxr ==0) Energy_table[idx] = Energy[c-1][lp + idxr].x + RowE[c];

                        else {

                            int max_energy=0;

                            if(idxr > 0)

                                for(signed int idxri=0;idxri<idxr;idxri++) {

                                    unsigned char* row = _imgEnergy.ptr<unsigned char>(min_index + lp + idxri);

                                    int ener = row[c];

                                    if(ener>max_energy)

                                        max_energy = ener;

                                }

                            if(idxr < 0)

                                for(signed int idxri=0;idxri<-idxr;idxri++) {

                                    unsigned char* row = _imgEnergy.ptr<unsigned char>(min_index + lp - idxri);

                                    int ener = row[c];

                                    if(ener>max_energy)

                                        max_energy = ener;

                                }

                            Energy_table[idx] = Energy[c-1][lp + idxr].x + max_energy / cost;

                        }

                    }

                    if(Energy_table[idx]<energy_min) {

                        energy_min = Energy_table[idx];

                        idxr_min = idxr;

                    }

                }

                Energy[c][lp].y = lp+idxr_min;

                Energy[c][lp].x = energy_min;

			}

		}

	}

	/*-----------------------------------------------------------------------------*/

    void SeamCarving::computeBackwardEnergy(std::vector<std::vector<cv::Point> >& Energy, int min_index, int max_index, int line)

	{

		if (_img.cols <= 2) return;

		int size_energy = max_index - min_index + 1;



		int mini_index = 0;

		int min_energy = INT_MAX;

		for (int lp = 0; lp < size_energy; lp++) {

			int mini = _medial_seam[line][_img.cols - 1];

			int maxi = _medial_seam[line + 1][_img.cols - 1];

			int en = Energy[_img.cols - 1][lp].x;

			if (en <= min_energy && (lp +min_index>=mini) && ((lp + min_index <= maxi))) {

				min_energy = en;

				mini_index = lp ;

			}

		}

		_carve_seam[line][_img.cols - 1] = mini_index+min_index;

		for (int c = _img.cols - 2; c >= 0; c--) {

			_carve_seam[line][c] = Energy[c + 1][mini_index].y + min_index;

			mini_index = Energy[c + 1][mini_index].y;

		}

	}

	/*-----------------------------------------------------------------------------*/

}



#endif
