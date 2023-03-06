#include <iostream>
#include "PerformancePlot.h"
#include <bits/stdc++.h>
#include <chrono>
#include <ctime> 

CustomData::CustomData() : PerformancePlot("./","Custom Data"){
    std::fprintf(this->pipehandle_,this->set_x_label_,"Iterations (n)");
    std::fprintf(this->pipehandle_,this->set_y_label_ , "Data");
}

CustomData::CustomData(std::string path , std::string title) : PerformancePlot(path,title){
    std::fprintf(this->pipehandle_,this->set_x_label_,"Iterations (n)");
    std::fprintf(this->pipehandle_,this->set_y_label_ , "Data");

}

void CustomData::Stamp(double data){
    this->y_values_.push_back(data);// Update Y value
    this->x_values_.push_back(this->iterations); //Update X value

    // buffer[x] = y
    std::fprintf(this->pipehandle_,
        this->cast_data_, //cmd
        this->iterations,// X Max *Start at 1*
        data); // y @ current x
    
    *csv << std::to_string(this->iterations) << std::to_string(data) << endrow;

    this->iterations++;
    
}

void CustomData::Stamp(){} //No need to implement, For override in FPS member function

void CustomData::StartRecord(){}//No need to implement, For override in FPS member function