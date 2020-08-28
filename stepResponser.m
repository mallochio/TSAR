function [y] = stepResponser(temp)
ytfinal = mean(temp(end-99:end));
t = linspace(0, size(temp(:,1),1) - 1, size(temp(:,1),1));

S1 = stepinfo(temp(:,1), t, ytfinal);
S2 = stepinfo(temp(:,2),  t, ytfinal);
%S3 = stepinfo(temp(:,3),  t, ytfinal);

meanrise = mean([S1.RiseTime, S2.RiseTime])%, S3.RiseTime])
meanset = mean([S1.SettlingTime, S2.SettlingTime])%,S3.SettlingTime])

%sprintf('Rise=', meanrise)
%sprintf('settling = ', meanset)
% %settling time - stays within epsilon of final value
% eps = 0.9;
% 
% level10 = (1-eps)*max(temp(:,:));
% level90 = eps*max(temp(:,:)); %outputs 3 values from max
% 
% point10 = [0,0,0];
% point90 = [0,0,0];
% %Rise time - Rise episodes to go from 10% to 90% of values
% for i = 1:size(temp,1)
%     for j = 1:size(temp,1)
%         if (temp(i,j) >= level10)
%             if point10(j) == 0
%                 point10(j) = i;
%             end
% 
%         elseif temp(i,j) >= level90
%             if point90(j) == 0
%                 point90(j) = i;
%             end   
%         end  
%     end
% end   
% 
% 
% t_r = point90 - point10; %set return values here
% t_s = 0;
y=0;
end