#include "Logger.h"

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

void logger(unsigned int i) {
	std::ofstream outfile;

	outfile.open("test.txt", std::ios_base::app);
	outfile << i << std::endl;
}

void logger(float f) {
	std::ofstream outfile;

	outfile.open("test.txt", std::ios_base::app);
	outfile << f << std::endl;
}

void logger(long l) {
	std::ofstream outfile;

	outfile.open("test.txt", std::ios_base::app);
	outfile << l << std::endl;
}

void logger(unsigned long long ul) {
	std::ofstream outfile;

	outfile.open("test.txt", std::ios_base::app);
	outfile << ul << std::endl;
}