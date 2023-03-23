#include <iostream>
#include <math.h>

using namespace std;

//Function Declarations
float Sigmoid(float _x);


//Main Script
int main()
{
    int NEpoch = 1000000; //Number of epoch
    float alpha = 0.6;//Learning rate

    //Trainning data
    float x1[4] = { 1.0, 0.0, 1.0, 0.0 };
    float x2[4] = { 1.0, 1.0, 0.0, 0.0 };
    float Yd5[4] = { 0.0, 1.0, 1.0, 0.0 };

    //Set inital weight and bias values (Set to random value in real thing)
    float w1_3 = 0.5; float w2_3 = 0.4;
    float w1_4 = 0.9; float w2_4 = 1.0;
    float w3_5 = -1.2; float w4_5 = 1.1;
    float Bias3 = 0.8; float Bias4 = -0.1; float Bias5 = 0.3;



    //Training loop
    for (int CEpoch = 0; CEpoch < NEpoch; CEpoch++) //CEpoch = Current Epoch 
    {
        float EpocSumError = 0; //Initate the termination condition value

        cout << "Epoch Number: " << CEpoch + 1 << endl; //Outputs current epoch

        //Begining of an Epoch
        for (int i = 0; i < 4; i++)
        {
            //====================================================================================================FEED FORWARDS
            float x3 = x1[i] * w1_3 + x2[i] * w2_3 + Bias3; //Calculate output for neuron 3
            float y3 = Sigmoid(x3); //Activation Function neuron 3

            float x4 = x1[i] * w1_4 + x2[i] * w2_4 + Bias4; //Calculate output for neuron 4
            float y4 = Sigmoid(x4); //Activation Function neuron 4

            float x5 = y3 * w3_5 + y4 * w4_5 + Bias5; //Calculate output for neuron 5
            float y5 = Sigmoid(x5); //Activation Function neuron 5

            //=====================================================================================================BACK PROPAGATION
            float e5 = Yd5[i] - y5; //Cost function for neuron 5 (to find error)

            float wCurrent3_5 = w3_5; //Saves the current w value before updating it
            float wCurrent4_5 = w4_5; //Saves the current w value before updating it

            //Neuron 5
            float Delta5 = y5 * (1 - y5) * e5;
            w3_5 = w3_5 + alpha * y3 * Delta5;
            w4_5 = w4_5 + alpha * y4 * Delta5;
            Bias5 = Bias5 + alpha * 1 * Delta5;

            //Neuron 3
            float Delta3 = y3 * (1 - y3) * Delta5 * wCurrent3_5;
            w1_3 = w1_3 + alpha * x1[i] * Delta3;
            w2_3 = w2_3 + alpha * x2[i] * Delta3;
            Bias3 = Bias3 + alpha * 1 * Delta3;

            //Neuron 4
            float Delta4 = y4 * (1 - y4) * Delta5 * wCurrent4_5;
            w1_4 = w1_4 + alpha * x1[i] * Delta4;
            w2_4 = w2_4 + alpha * x2[i] * Delta4;
            Bias4 = Bias4 + alpha * 1 * Delta4;

            //====================================================================================ERROR CHECKING (Checks if new wieghts and biases prodcues the correct outputs)
            float tx3 = x1[i] * w1_3 + x2[i] * w2_3 + Bias3; 
            float ty3 = Sigmoid(tx3);
            float tx4 = x1[i] * w1_4 + x2[i] * w2_4 + Bias4;
            float ty4 = Sigmoid(tx4);
            float tx5 = ty3 * w3_5 + ty4 * w4_5 + Bias5;
            float ty5 = Sigmoid(tx5);
            float te5 = Yd5[i] - ty5; //Error of neuron 5 output (final output)

            EpocSumError = EpocSumError + pow(te5, 2); //Squared Error

            //-----------!Debug Code (Can be removed later)!
            /*
            cout << "\n\n\n\n\n\ni = " << i + 1 << endl;
            cout << "\ny(3) = " << y3 << endl;
            cout << "y(4) = " << y4 << endl;
            cout << "y(5) = " << y5 << endl;
            cout << "e(5) = " << e5 << endl;
            cout << "\ndelta(5) = " << Delta5 << endl;
            cout << "w(3,5) = " << w3_5 << endl;
            cout << "w(4,5) = " << w4_5 << endl;
            cout << "Bias(5) = " << Bias5 << endl;
            cout << "\ndelta(3) = " << Delta3 << endl;
            cout << "w(1,3) = " << w1_3 << endl;
            cout << "w(2,3) = " << w2_3 << endl;
            cout << "Bias(3) = " << Bias3 << endl;
            cout << "\ndelta(4) = " << Delta4 << endl;
            cout << "w(1,4) = " << w1_4 << endl;
            cout << "w(2,4) = " << w2_4 << endl;
            cout << "Bias(4) = " << Bias4 << endl;
            cout << "\nte(5) = " << te5 << endl;
            cout << "EpocSumError = " << EpocSumError << endl;
            */
        }

        //If squard error in Epoch is acceptable (correct weights and biases) break out of trainning loop
        if (EpocSumError < 0.001)
        {
            break;
        }
    }

    //Output Results
    cout << "\nWeights:" << endl;
    cout << "w(1,3) = " << w1_3 << endl;
    cout << "w(2,3) = " << w2_3 << endl;
    cout << "w(1,4) = " << w1_4 << endl;
    cout << "w(2,4) = " << w2_4 << endl;
    cout << "w(3_5) = " << w3_5 << endl;
    cout << "w(4,5) = " << w4_5 << endl;
    cout << "\nBiases:" << endl;
    cout << "Bias(3) = " << Bias3 << endl;
    cout << "Bias(4) = " << Bias4 << endl;
    cout << "Bias(5) = " << Bias5 << endl;
}



//Activation Functions
float Sigmoid(float _x)
{
    float e = 2.71828; //Euler's Number
    return 1 / (1 + pow(e, -(_x)));
}

//Max function can be used for ReUL
//Derivative is calculas (recap before coding)

//Use square sum error to train error (at the moment it is only used to terminate program when correct errors are found)