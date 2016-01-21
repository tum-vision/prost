#ifndef PDSOLVER_EXCEPTION_HPP_
#define PDSOLVER_EXCEPTION_HPP_


#include <cstdlib>

using namespace std;


///
/// \brief Used to throw and catch exceptions within the ShapeAnalyzer.
/// \details Exceptions that inherit from Error are expected to be catched, or
/// it is generally possible to catch and handle them. Every namespace has its
/// own Error class to allow more specific error handling.
/// @author Emanuel Laude 
///

class PDSolverException : public exception {
public:
    /// \brief Standard constructor.
    /// \details If possible use the specific constructor PDSolverException(const std::string&) and
    /// give a message indicating the kind of problem.
    PDSolverException() : what_("An error occured in the solver.") {}

    /// \brief Specific constructor.
    /// \details The given string will be shown in an error dialog. Please make it meaningful.
    PDSolverException(const std::string& str) {
        what_ = strdup(str.c_str());
    }

    

    /// \brief Returns the what_-message indicating the kind of problem.
    virtual const char* what() const throw() { return what_; }
private:
    const char* what_;
};

#endif



