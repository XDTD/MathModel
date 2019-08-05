%% Q2
ri = [5 28 21 23 25]/100;
qi = [0 2.5 1.5 5.5 2.6 ]/100;
pi = [0 1 2 4.5 6.5]/100;
figure
fff=[];
k=0;
while k<0.3
    ri_pi = [.05 .27 .19 .185 .185];
    x0 = [0 0 0 0 0];
    fun=@(x)(qi.*x);
    A = -ri_pi;
    b = [-k];
    Aeq=[1 ,1.01 ,1.02, 1.045 ,1.065];
    beq=1;
    vlb=[0,0,0,0,0];
    vub=[];
    [x,~,val,exitflag]=fminimax(fun,x0,A,b,Aeq,beq,vlb,vub);
    fff=[fff,exitflag];
    Q =val;
    plot(k,Q,'.')
    hold on
    k = k +0.001;
    if exitflag<0
        break;
    end
end
title('风险改变图')
%% Q3
ri = [5 28 21 23 25]/100;
qi = [0 2.5 1.5 5.5 2.6 ]/100;
pi = [0 1 2 4.5 6.5]/100;
figure
fff=[];
s=0;
res1=[];
res2=[];
while s<1
    ri_pi = [.05 .27 .19 .185 .185];
    x0 = [0 0 0 0 0];
    fun=@(x)(max(qi.*x)-(1-s)*ri_pi*x');
    A=[];
    b=[];
    Aeq=[1 ,1.01 ,1.02, 1.045 ,1.065];
    beq=1;
    vlb=[0,0,0,0,0];
    vub=[];
    [X,val,exitflag]=fmincon(fun,x0,A,b,Aeq,beq,vlb,vub);
    fff=[fff,exitflag];
    Q =val;
    res1=[res1,ri_pi*X'];
    res2=[res2,max(qi.*X)];
    if exitflag~=1
        break;
    end
    s = s +0.001;
end
figure
hold on
plot(0:0.001:s,res1,'.');
title('收益图')
figure
hold on
plot(0:0.001:s,res2,'.');
title('风险图')