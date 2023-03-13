#include <iostream>
#include "PerformancePlot.h"
#include <bits/stdc++.h>
#include <chrono>
#include <ctime> 

FPS::FPS() : PerformancePlot("./","Performance Plotting") {
    std::fprintf(this->pipehandle_,this->set_x_label_,"Iterations (n)");
    std::fprintf(this->pipehandle_,this->set_y_label_ , "FPS");
}

FPS::FPS(std::string path , std::string title) : PerformancePlot(path,title){
    std::fprintf(this->pipehandle_,this->set_x_label_,"Iterations (n)");
    std::fprintf(this->pipehandle_,this->set_y_label_ , "FPS");
}

void FPS::StartRecord(){
    this->start_ = clock();
}

void FPS::Stamp(){
    this->end_ = clock();
    this->y_values_.push_back(1 / (double(this->end_ - this->start_) / double(CLOCKS_PER_SEC)));// Update Y value
    this->x_values_.push_back(this->iterations); //Update X value

    // buffer[x] = y
    std::fprintf(this->pipehandle_,
        this->cast_data_, //cmd
        this->iterations,// X Max *Start at 1*
        1 / (double(this->end_ - this->start_) / double(CLOCKS_PER_SEC))); // y @ current x
    
    *csv << std::to_string(this->iterations) << std::to_string(double(this->end_ - this->start_) / double(CLOCKS_PER_SEC)) << endrow;



    this->iterations++;
    
}

void FPS::Stamp(double data){}//No need to implement, For override in CustomData member function