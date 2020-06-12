#include "Logger.h"

#include <fstream>

void logger(string text) {
	std::ofstream outfile;

	outfile.open("test.txt", std::ios_base::app);
	outfile << text << std::endl;
}

void logger(int i) {
	std::ofstream outfile;

	outfile.open("test.txt", std::ios_base::app);
	outfile << i << std::endl;
}

void logger(float f) {
	std::ofstream outfile;

	outfile.open("test.txt", std::ios_base::app);
	outfile << f << std::endl;
}