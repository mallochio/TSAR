function line = addToCostPlot( rewardtrajectories, color, fighan, conf )

figure(fighan);
hold on
if(size(rewardtrajectories,2) > 1)
     SEM = std(rewardtrajectories)/sqrt(size(rewardtrajectories,1)); %Standard Error
     ts = tinv([(1-conf)  conf],size(rewardtrajectories,1)-1); % T-Score
     CI = [mean(rewardtrajectories,1); mean(rewardtrajectories,1)] + (ts.'*SEM); %lin
     x=1:size(rewardtrajectories,2);     %#initialize x array
     
     y1=CI(1,:);
     y2=CI(2,:);
     X=[x,fliplr(x)];                %#create continuous x value array sizefor plotting
     Y=[y1,fliplr(y2)];              %#create y values for out and then back
     a = fill(X,Y,color);          %#plot filled area
     set(a,'EdgeColor','None');
     alpha(a,0.35)
end

line = plot(1:size(rewardtrajectories,2),mean(rewardtrajectories,1),'Color',color,'LineWidth',2);
hold off

legend('1x', '', '10x', '','100x')
grid off
title('Returns Chart')
xlabel('Episodes')
ylabel('Normalized Returns')
end