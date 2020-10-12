#pragma once

#define FE_0(action)
#define FE_1(action, x, ...) action(x)
#define FE_2(action, x, ...) action(x),FE_1(action, __VA_ARGS__)
#define FE_3(action, x, ...) action(x),FE_2(action, __VA_ARGS__)
#define FE_4(action, x, ...) action(x),FE_3(action, __VA_ARGS__)
#define FE_5(action, x, ...) action(x),FE_4(action, __VA_ARGS__)

#define GET_MACRO(_0, _1, _2, _3, _4, _5, NAME, ...) NAME
#define FOR_EACH(action, ...) \
    GET_MACRO(_0, __VA_ARGS__, FE_5, FE_4, FE_3, FE_2, FE_1, FE_0)(action, __VA_ARGS__)

#define EAT(...)
#define EAT_TYPE(x) EAT x
#define ID(x) x
#define STRIP_TYPE_BRACKET(x) ID x

#define PARAM_LIST(...) FOR_EACH(STRIP_TYPE_BRACKET, __VA_ARGS__)
#define PARAM_NAMES(...) FOR_EACH(EAT_TYPE, __VA_ARGS__)

#define XPU_STRINGIZE_I(val) #val
#define XPU_STRINGIZE(val) XPU_STRINGIZE_I(val)