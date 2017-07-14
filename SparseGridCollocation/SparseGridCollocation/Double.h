#pragma once

class Double
{
private:
	double d;

	static double MatLabRounding(double d)
	{
		//return round(d * 1000000000000000) / 1000000000000000;
		return round(d * 10000000000000) / 10000000000000;
	};

public:
	Double(double v) : d(v) {}


	friend Double operator+(Double &a, Double &b)
	{
		return MatLabRounding(a.value() + b.value());
	};

	friend Double operator-(Double &a, Double &b)
	{
		return MatLabRounding(a.value() - b.value());
	};

	friend Double operator*(Double &a, Double &b)
	{
		return MatLabRounding(a.value() * b.value());
	};

	Double& operator=(Double &a)
	{
		d = MatLabRounding(a.value());
		return *this;
	};

	double value() const { return d; }
};
