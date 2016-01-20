#ifndef FACTORY_HPP_
#define FACTORY_HPP_

#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

using namespace std;

///
/// \brief Variadic template singleton factory class for creating instances of concrete CustomTab, CustomContextMenuItem and other classes.
///
/// \details Concrete classes are derived from an abstract superclass for example ColorEigenfunctionContextMenuItem is derived from the abstract class ContextMenuItem. Moreover concrete classes have to provide a public constructor for which the signature coincides with the type sequence given as the second template argument tuple.
/// @tparam class T. Abstract class for which the factory should be defined (e.g. ContextMenuItem)
/// @tparam typename... Args. Argument signature of the contructors of type T.
///
/// \author Thomas Möllenhoff and Emanuel Laude
///
template<class T, class... Args>
class Factory {
public:
    /// \brief Destructor. Clears the internal data structures.
    ~Factory() {
        create_funs_.clear();
    }
    
    /// \brief Returns the unique instance of Factory<T>.
    /// \details This is the only way to obtain the Factory object for type T.
    static Factory<T, Args...>* GetInstance() {
        static Factory<T, Args...> instance;
        return &instance;
    }

    /// \brief Registers concrete class derived from type T in the internal data structures of Factory.
    /// @tparam class C. The concrete class to be registered.
    /// @param const string& identifier. A unique string that is used as a key to obtain an instance via create() afterwards.
    /// @param const string& label. A label that can for example serve as menu path in case of CustomContextMenuItem
    template<class C>
    void Register(const string& identifier, const function<T*(Args...)>& create_fun) {
        create_funs_.insert(pair<string, function<T*(Args...)>>(identifier, create_fun));
    }
    
    /// \brief Obtain a new instance of a class derived from T using the unique identifier of the class and a std::tuple of arguments required to instantiate the class. Throws an exception of type std::out_of_range if identifier is not present.
    /// @param const string& identifier. identifier of the class.
    /// @param std::tuple. Sequence of of arguments for the create function.
    T* Create(const string& identifier, Args... args) {
        // execute create function pointer given the arguments in tuple args.
        return create_funs_.at(identifier)(args...);
    }
private:
    /// \brief Private constructor.
    /// \details Use getInstance() to obtain the unique instance of this singleton.
    Factory() {
    }
    
    /// \brief Private copy constructor.
    /// \details Use getInstance() to obtain the unique instance of this singleton.
    Factory(const Factory&) {
    }
    
    /// \brief Private assignment operator.
    /// \details Use getInstance() to obtain the unique instance of this singleton.
    Factory& operator=(const Factory&) {
        return *this;
    }
    
    /// \brief Map containing all identifier plus create function pointer pairs.
    unordered_map<string, function<T*(Args...)>> create_funs_;
};

#endif