#include <iostream>
#include "PerformancePlot.h"
#include <bits/stdc++.h>
#include <chrono>
#include <ctime> 

PerformancePlot::PerformancePlot(std::string path , std::string title){
    
    //###################TIME STAMP###################
    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::string current = std::ctime(&end_time);
    current.erase(current.length() - 6); // Delete '$'\n' in string
    //###################TIME STAMP###################


    //###################PLOT Title Initial###################
    this->plot_title_ = new char[title.size()]; //Create object from null pointer
    strcpy(this->plot_title_ , title.c_str());// Copy String buffer
    //###################PLOT Title Initial###################

    //###################Graph Save Initial###################
    this->path_save_graph_ = new char[(path + current + title + ".png").size()];//Create object from null pointer
    strcpy(this->path_save_graph_ , (path + current + title + ".png").c_str());// Copy String buffer
    //###################Graph Save Initial###################

    this->csv = new csvfile((path + current + title + ".csv").c_str());// Initial CSV File
    
    std::fprintf(this->pipehandle_,this->create_array_);
    std::fprintf(this->pipehandle_,this->unset_mouse_);
    std::fprintf(this->pipehandle_,this->set_font_);
    std::fprintf(this->pipehandle_,this->set_grid_);
}

void PerformancePlot::End() const{
    std::fflush(this->pipehandle_);//Flush pipeline Buffer
    std::fclose(this->pipehandle_);//Close File
    
}

void PerformancePlot::Show(){
    std::fprintf(this->pipehandle_ , this->set_xrange_,this->iterations);
    std::fprintf(this->pipehandle_ , this->set_yrange_, *std::max_element(this->y_values_.begin() , this->y_values_.end()));// Y max
    std::fprintf(this->pipehandle_ , this->plot_);
    std::fprintf(this->pipehandle_ , this->plot_prop_ , this->plot_title_);
}

void PerformancePlot::Save(){
    //Save Graph
    std::fprintf(this->pipehandle_ , this->set_output_,this->path_save_graph_);
    std::fprintf(this->pipehandle_ , this->replot_ );

}

void PerformancePlot::Clear(){
    this->y_values_.clear();//Clear Buffer y
    this->x_values_.clear();//Clear Buffer x
    iterations = 1;//Iteration resets to 1 Because GNU index starts at 1
}

extern "C" {
    PerformancePlot* PerformanceFPS(){ return new FPS(); }
    PerformancePlot* PerformanceCustomData(){ return new CustomData(); }

    void Stamp(CustomData *PerformanceCustomData , double data)
    {
        PerformanceCustomData->Stamp(data);
    }

    void Show(CustomData *PerformanceCustomData)
    {
        PerformanceCustomData->Show();
    }

    void Save(CustomData *PerformanceCustomData)
    {
        PerformanceCustomData->Save();
    }

    void End(CustomData *PerformanceCustomData)
    {
        PerformanceCustomData->End();
    }

}
