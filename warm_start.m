function warm_start()
    x = [-1 -1 1 1];
    y = [-1 1 -1 1];
    theta = [-1 -.5 .3 -.4];
    gama = 1;
    distance = zeros(length(x),length(x));
    for i=1:length(x)
       for j=1:length(x)
            distance(i,j)=(x(i)-x(j))^2+(y(i)-y(j))^2;
       end
    end
    res=f(theta,gama,distance);
    x=[-1 1];
    y=[-1 1];
    [X,Y]=meshgrid(x,y);
    Z=zeros(size(X));
    Z(1,:)=res(1:2);
    Z(2,:)=res(3:4);
    contour3(X,Y,Z);
end

% 
% function res=f(theta,gama,x,y,idx)
%     res = 0;
%     for i=1:length(x)
%         res = theta(i)*exp(-gama*((x(idx)-x(i)^2)+(y(idx)-y(i))^2))+res;
%     end
% 
% end

function res=f(theta,gama,distance)
    res = theta*exp(-gama*distance');
end