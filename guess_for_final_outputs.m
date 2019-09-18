function GUESS=project(A,B,guess)
% Book 3, is the original training data with the each base stat divided by the
% total.
% Book 4 is similar but for the test data
% Book 6 is the results from the python code converted into an Excel file.
A=xlsread('Book3')
B=xlsread('Book4')
guess=xlsread('Book6')
GUESS=guess;
for i=1:201
    v1=A(find(A(:,1)==GUESS(i,1)+1),:);
    v2=A(find(A(:,1)==GUESS(i,1)-1),:);
    disp(v1)
    disp(v2)
    CLOSEMAT=[v1;v2];
    CLOSEMAT2=CLOSEMAT(:,10:15);
    sim=zeros(size(CLOSEMAT,1),1);
    for r=1:size(CLOSEMAT2,1)
        error=abs(CLOSEMAT2(r,:)-B(i,9:14));
        disp(error)
        if (error'*error)<=1
            sim(r)=1;
        end
    end
    disp(sim)
    if sum(sim)==0 && size(CLOSEMAT,1)>1 && CLOSEMAT(1,2)==CLOSEMAT(2,2) && (CLOSEMAT(2,9)<=B(i,8) || B(i,8)<=CLOSEMAT(1,9))
        GUESS(i,2)=CLOSEMAT(1,2);
    else if sum(sim)==2
            if CLOSEMAT(1,2)==CLOSEMAT(2,2)
                GUESS(i,2)=CLOSEMAT(1,2);
            else
                if CLOSEMAT(2,9)<=B(i,8) && B(i,8)>=CLOSEMAT(1,9)
                    GUESS(i,2)=CLOSEMAT(2,2);
                else if CLOSEMAT(2,9)>=B(i,8) && B(i,8)<=CLOSEMAT(1,9)
                        GUESS(i,2)=CLOSEMAT(1,2);
                    else if CLOSEMAT2(1,:)*B(i,9:14)'>CLOSEMAT2(2,:)*B(i,9:14)'
                            GUESS(i,2)=CLOSEMAT(1,2);
                        else
                            GUESS(i,2)=CLOSEMAT(2,2);
                        end
                    end
                end
            end
        else if sum(sim)==1
                if sim(1)==1
                    if size(v1,1)~=0 && CLOSEMAT(1,9)>=B(i,8)
                        GUESS(i,2)=CLOSEMAT(1,2);
                    else if size(v1,1)==0 && CLOSEMAT(1,9)<=B(i,8)
                        GUESS(i,2)=CLOSEMAT(1,2);
                        end
                    end
                else
                    if CLOSEMAT(2,9)<=B(i,8)
                        GUESS(i,2)=CLOSEMAT(2,2);
                    end
                end
            end
        end
    end
end
T=table(ans(:,1),ans(:,2));
T.properties.VariableNames={'Number','Type'};
T.Properties.VariableNames={'Number','Type'};
writetable(T,'Bestguess.csv')
end