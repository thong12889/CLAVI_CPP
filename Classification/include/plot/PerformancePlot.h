#include <iostream>
#include <vector>
#include <fstream>
#include "csvfile.h"

#ifndef PERFORMANCEPLOT_H
#define PERFORMANCEPLOT_H

class PerformancePlot{
    public:
        PerformancePlot(std::string path , std::string title);
        virtual void StartRecord() = 0;
        virtual void Stamp() = 0;
        virtual void Stamp(double) = 0;
        void End() const;
        void Show();
        void Save();
        void Clear();


        
    private:
        //Realtime Plot
        //Title
        char* plot_title_;
        char *path_save_graph_;

        //cmd pipeline
        

        char *create_array_ = "array buffer[10000]\n";
        char *end_line_ = "\n";
        char *set_autoscale_ = "set autoscale\n";
        char *set_xrange_ = "set xrange[0:%lf]\n";
        char *set_yrange_ = "set yrange[0:%lf]\n";
        char *unset_mouse_ = "unset mouse\n";
        char *set_mouse_ = "set mouse\n";
        char *set_grid_ = "set grid\n";
        
        char *plot_ = "plot ";
        char *plot_prop_ = "buffer w l lt 7 lc 6  title \"%s\"\n";
        char *set_font_ = "set key font 'Helvetica,10'\n";

        char *set_output_ = "set terminal png size 1920,1080 enhanced font 'Helvetica,15'; set output \"%s\" \n";
        char *replot_ = "replot\n";

        
    protected:
        char *set_x_label_ = "set xlabel '%s'\n";
        char *set_y_label_ = "set ylabel '%s'\n";

        std::FILE* pipehandle_=popen("gnuplot -persistent","w");
        char *cast_data_ = "buffer[%lf] = %lf\n";

        //Save File txt
        csvfile *csv;

        std::vector<double> x_values_;
        std::vector<double> y_values_;
        double iterations = 1;

        
};

class FPS : public PerformancePlot{
    public:
        FPS();
        FPS(std::string path , std::string title);
        void StartRecord();
        void Stamp();
        void Stamp(double data);
    private:
        clock_t start_ , end_;

};

class CustomData : public PerformancePlot{
    public:
        CustomData();
        CustomData(std::string path , std::string title);
        void StartRecord();
        void Stamp();
        void Stamp(double data);
    private:
};



#endif