#pragma once

#include <string>
#include <fstream>

using namespace std;

void logger(string text);
void logger(int i);
void logger(unsigned int i);
void logger(float f);
void logger(long l);
void logger(unsigned long long ul);

template<typename T>
void logger(string text, T t) {
	std::ofstream outfile;

	outfile.open("test.txt", std::ios_base::app);
	outfile << text << ": " << t << std::endl;
}