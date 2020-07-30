clear;
%Population na CPU
%file = csvread('Your path to project\results\algorithm_results_cpu.csv');

%Population na GPU
file = csvread('Your path to project\results\algorithm_results_gpu.csv');

how_many_members = file(1,1);
Which_function = file(1,3);
Show_2D = 1;
Show_3Din2D = 1;
Show_3D = 1;
Show_optimum = 1;
resolution = 1;
frame_interval = 0.2;

X_min = file(size(file, 1)-1,1);
X_max = file(size(file, 1)-1,2);
Y_min = file(size(file, 1),1);
Y_max = file(size(file, 1),2);

How_many_generations = file(1,2);

string = ["Amount of members = ", how_many_members, "Amount of generations = ",  How_many_generations, "Function nr ", Which_function];
disp(string);

if(Show_2D == 1)
    figure('Name', 'Members-2D');
    pause(1);
    for i = 0:(How_many_generations-1)
        x_member = file(2+i,1:2:end);
        y_member = file(2+i,2:2:end);
        plot(x_member, y_member, 'ro')
        xlim([X_min X_max])
        ylim([Y_min Y_max])
        pause(frame_interval)
    end
end

if(Show_3Din2D == 1)
    figure('Name', 'Members-3Din2D');
    [X,Y] = meshgrid(X_min:resolution:X_max, Y_min:resolution:Y_max);

    if(Which_function==1)
        Z = (sin(X) + cos(Y));
    elseif (Which_function==2)
        Z = (X.^2 + Y.^2);
    elseif(Which_function==3)
        Z = (Y.*sin(X) - X.*cos(Y));
    elseif(Which_function==4)
        Z = (sin(X+Y) + (X-Y).^2 - 1.5*X + 2.5*Y + 1);
    elseif(Which_function==5)
        Z = -(Y+47).*sin(sqrt(abs((X./2) + (Y+47)))) - X.*sin(sqrt(abs(X-(Y+47))));
    elseif(Which_function==6)
        Z = 418.9829*2 - (X.*sin(sqrt(abs(X))) + Y.*sin(sqrt(abs(Y))));
    end

    pause(1);
    
     for i = 0:(How_many_generations-1)
        surf(X, Y, Z, 'edgecolor', 'none');
        view(2)
        hold on
        x_member = file(2+i,1:2:end);
        y_member = file(2+i,2:2:end);

        if(Which_function ==1)
            z_member = (sin(x_member) + cos(y_member));
        elseif(Which_function == 2)
            z_member = (x_member.^2 +y_member.^2);
        elseif(Which_function ==3)
            z_member = (y_member.*sin(x_member) - x_member.*cos(y_member));
         elseif(Which_function==4)
            z_member = (sin(x_member+y_member) + (x_member-y_member).^2 - 1.5*x_member + 2.5*y_member + 1);
        elseif(Which_function==5)
            z_member = -(y_member+47).*sin(sqrt(abs((x_member./2) + (y_member+47)))) - x_member.*sin(sqrt(abs(x_member-(y_member+47))));
        elseif(Which_function==6)
            z_member = 418.9829*2 - (x_member.*sin(sqrt(abs(x_member))) + y_member.*sin(sqrt(abs(y_member))));
        end

        plot3(x_member, y_member, z_member, 'rx')
        xlim([X_min X_max])
        ylim([Y_min Y_max])
        hold off
        pause(frame_interval)
    end
    
    if(Show_optimum==1)
        surf(X, Y, Z, 'edgecolor', 'none');
        view(2)
        hold on
        x_opt = file(size(file,1)-2,1);
        y_opt = file(size(file,1)-2,2);

        if(Which_function ==1)
                z_opt = (sin(x_opt) + cos(y_opt));
        elseif(Which_function == 2)
                z_opt = (x_opt.^2 +y_opt.^2);
        elseif(Which_function ==3)
                z_opt = (y_opt.*sin(x_opt) - x_opt.*cos(y_opt));
        elseif(Which_function==4)
                z_opt = (sin(x_opt+y_opt) + (x_opt-y_opt).^2 - 1.5*x_opt + 2.5*y_opt + 1);
        elseif(Which_function==5)
                z_opt = -(y_opt+47).*sin(sqrt(abs((x_opt./2) + (y_opt+47)))) - x_opt.*sin(sqrt(abs(x_opt-(y_opt+47))));
       elseif(Which_function==6)
            z_opt = 418.9829*2 - (x_opt.*sin(sqrt(abs(x_opt))) + y_opt.*sin(sqrt(abs(y_opt))));
        
        end

        plot3(x_opt, y_opt, z_opt, 'w+')
        xlim([X_min X_max])
        ylim([Y_min Y_max])
        hold off
    end
end

if(Show_3D == 1)
    figure('Name', 'Members-3D');
    [X,Y] = meshgrid(X_min:resolution:X_max, Y_min:resolution:Y_max);

    if(Which_function==1)
        Z = (sin(X) + cos(Y));
    elseif (Which_function==2)
        Z = (X.^2 + Y.^2);
    elseif(Which_function==3)
        Z = (Y.*sin(X) - X.*cos(Y));
    elseif(Which_function==4)
        Z = (sin(X+Y) + (X-Y).^2 - 1.5*X + 2.5*Y + 1);
    elseif(Which_function==5)
        Z = -(Y+47).*sin(sqrt(abs((X./2) + (Y+47)))) - X.*sin(sqrt(abs(X-(Y+47))));
    elseif(Which_function==6)
        Z = 418.9829*2 - (X.*sin(sqrt(abs(X))) + Y.*sin(sqrt(abs(Y))));
    end

    pause(1);
    
    for i = 0:(How_many_generations-1)
        surf(X, Y, Z, 'edgecolor', 'none');
        hold on
        x_member = file(2+i,1:2:end);
        y_member = file(2+i,2:2:end);

        if(Which_function ==1)
            z_member = (sin(x_member) + cos(y_member));
        elseif(Which_function == 2)
            z_member = (x_member.^2 +y_member.^2);
        elseif(Which_function ==3)
            z_member = (y_member.*sin(x_member) - x_member.*cos(y_member));
         elseif(Which_function==4)
            z_member = (sin(x_member+y_member) + (x_member-y_member).^2 - 1.5*x_member + 2.5*y_member + 1);
       elseif(Which_function==5)
            z_member = -(y_member+47).*sin(sqrt(abs((x_member./2) + (y_member+47)))) - x_member.*sin(sqrt(abs(x_member-(y_member+47))));
       elseif(Which_function==6)
            z_member = 418.9829*2 - (x_member.*sin(sqrt(abs(x_member))) + y_member.*sin(sqrt(abs(y_member))));
        end

        plot3(x_member, y_member, z_member, 'rx')
        xlim([X_min X_max])
        ylim([Y_min Y_max])
        hold off
        pause(frame_interval)
    end
    
    if(Show_optimum==1)
        surf(X, Y, Z, 'edgecolor', 'none');
        hold on
        x_opt = file(size(file,1)-2,1);
        y_opt = file(size(file,1)-2,2);

        if(Which_function ==1)
                z_opt = (sin(x_opt) + cos(y_opt));
        elseif(Which_function == 2)
                z_opt = (x_opt.^2 +y_opt.^2);
        elseif(Which_function ==3)
                z_opt = (y_opt.*sin(x_opt) - x_opt.*cos(y_opt));
        elseif(Which_function==4)
                z_opt = (sin(x_opt+y_opt) + (x_opt-y_opt).^2 - 1.5*x_opt + 2.5*y_opt + 1);
       elseif(Which_function==5)
                z_opt = -(y_opt+47).*sin(sqrt(abs((x_opt./2) + (y_opt+47)))) - x_opt.*sin(sqrt(abs(x_opt-(y_opt+47))));
       elseif(Which_function==6)
            z_opt = 418.9829*2 - (x_opt.*sin(sqrt(abs(x_opt))) + y_opt.*sin(sqrt(abs(y_opt))));
        
        end

        plot3(x_opt, y_opt, z_opt, 'w+')
        xlim([X_min X_max])
        ylim([Y_min Y_max])
        hold off
    end
end


