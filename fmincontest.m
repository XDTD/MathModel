function fmincontest()
clc
clear
close all
fun=@(x)max(x);
x0=rand(2,1);
A=[];
b=[];
Aeq=[];
beq=[];
vlb=[0,0];
vub=[];
exitflag=1;
[x,fval,exitflag]=fmincon(fun,x0,A,b,Aeq,beq,vlb,vub)

end

