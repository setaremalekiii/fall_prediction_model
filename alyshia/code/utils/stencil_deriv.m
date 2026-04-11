function dx = stencil_deriv(dt, x)
    dx = zeros(size(x));
    dx(1) = -25/12*x(1) + 4*x(2) - 3*x(3) + 4/3*x(4) - 1/4*x(5);
    dx(2) = -25/12*x(2) + 4*x(3) - 3*x(4) + 4/3*x(5) - 1/4*x(6);
    for i = 3:(length(x)-2)
        dx(i) = 1/12*x(i-2) - 2/3*x(i-1) + 2/3*x(i+1) - 1/12*x(i+2);
    end
    dx(end) = 25/12*x(end) - 4*x(end-1) + 3*x(end-2) - 4/3*x(end-3) + 1/4*x(end-4);
    dx(end-1) = 25/12*x(end-1) - 4*x(end-2) + 3*x(end-3) - 4/3*x(end-4) + 1/4*x(end-5);
    dx = dx / dt;
end
