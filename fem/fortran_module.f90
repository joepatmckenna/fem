module fortran_module
  ! fortran_module description
  !$ use omp_lib
  implicit none

  public :: num_threads
  public :: simulate_time_series
  public :: simulate_mutations
  public :: discrete_fit
  public :: discrete_fit_i

contains

  function num_threads() result (n)
    ! Return the number of threads that the package can see.
    ! If the return value is 1 but there are multiple cpus,
    ! then OpenMP is not installed or not properly configured
    implicit none
    integer n
    n = 1
    !$ n = omp_get_max_threads()
  end function num_threads

  subroutine multiindices(n, m, degs, n_deg, n_idx, n_idx_by_deg, idx_ptr, idx, stratifier)
    implicit none

    integer, intent(in) :: n, n_deg, n_idx, m(n), degs(n_deg), n_idx_by_deg(n_deg)
    integer, intent(out) :: idx_ptr(:), idx(:), stratifier(:)
    integer, allocatable :: var(:)
    integer i, i1, i2, j, k

    idx_ptr(1) = 0
    i = 2
    do j = 1, n_deg
       do k = 1, n_idx_by_deg(j)
          idx_ptr(i) = idx_ptr(i-1) + degs(j)
          i = i + 1
       end do
    end do
    ! print*, 'idx_ptr', idx_ptr

    i = 1
    do j = 1, n_deg
       allocate(var(degs(j)))
       ! var = (/ (i2, i=12, degs(j)) /)
       do i2 = 1, degs(j)
          var(i2) = i2
       end do
       idx(idx_ptr(i)+1:idx_ptr(i+1)) = var
       i = i + 1
       ! do k = 1, n_idx_by_deg(degs(j))-1
       do k = 1, n_idx_by_deg(j)-1
          i1 = degs(j)
          var(i1) = var(i1) + 1
          do while (var(i1) + degs(j) > i1 + n)
             i1 = i1 - 1
             var(i1) = var(i1) + 1
          end do
          ! var(i1+1:) = (/ (var(i2-1)+1, i2=i1+1, degs(j)) /)
          do i2 = i1+1, degs(j)
             var(i2) = var(i2-1)+1
          end do
          idx(idx_ptr(i)+1:idx_ptr(i+1)) = var
          i = i + 1
       end do
       deallocate(var)
    end do
    ! print*, 'idx', idx

    stratifier = 1
    do i = 2, n_idx
       do j = idx_ptr(i-1)+1, idx_ptr(i)
          stratifier(i) = stratifier(i) * m(idx(j))
       end do
       stratifier(i) = stratifier(i) + stratifier(i-1)
    end do
    ! print*, 'stratifier', stratifier

  end subroutine multiindices

  subroutine simulate_time_series(w, m_sum, n_s, n, m, l, degs, n_deg, x)
    ! simulate_time_series description

    real*8, intent(in) :: w(m_sum, n_s)
    integer, intent(in) :: m_sum, n_s, n, n_deg, l, m(n), degs(n_deg)
    integer, intent(out) :: x(n,l)
    !f2py integer, intent(hide), depend(w) :: m_sum=shape(w,0)
    !f2py integer, intent(hide), depend(w) :: n_s=shape(w,1)
    !f2py integer, intent(hide), depend(m) :: n=len(m)
    !f2py integer, intent(hide), depend(degs) :: n_deg=len(degs)

    integer, allocatable :: bc(:), idx_ptr(:), idx(:), stratifier(:), s(:,:)
    integer t, i, i1, i2, j, max_deg, n_idx, n_idx_by_deg(n_deg), m_cumsum(n+1)
    real*8 randr(n, l), wrk(m_sum)

    ! print*, 'passed to fortran'

    call random_seed()
    call random_number(randr)

    max_deg = degs(n_deg)
    allocate(bc(max_deg))
    bc(1) = n
    do i = 2, max_deg
       bc(i) = bc(i-1)*(n-i+1)/i
    end do
    n_idx_by_deg = (/ (bc(degs(i)), i=1, n_deg) /)
    n_idx = sum(n_idx_by_deg)
    m_cumsum(1) = 0
    do i = 1, n
       m_cumsum(i+1) = m_cumsum(i) + m(i)
    end do
    ! print*, 'm_cumsum', m_cumsum

    allocate(idx_ptr(n_idx+1))
    allocate(idx(sum(n_idx_by_deg*degs)))
    allocate(stratifier(n_idx))

    call multiindices(n, m, degs, n_deg, n_idx, n_idx_by_deg, idx_ptr, idx, stratifier)

    ! random initial condition
    do i = 1, n
       x(i, 1) = mod(int((10*m(i)+1)*randr(i, 1)), m(i))
    end do
    allocate(s(n_idx, l))
    do i = 1, n_idx
       s(i, 1) = x(idx(idx_ptr(i)+1), 1)
       do j = idx_ptr(i)+2, idx_ptr(i+1)
          s(i, 1) = s(i, 1) * m(idx(j)) + x(idx(j), 1)
       end do
       s(i, 1) = s(i, 1) + stratifier(i)
    end do
    ! print*, 'initial x', x(:, 1)

    do t = 2, l

       ! print*, 't', t

       ! energies
       wrk = 0
       do i = 1, n_idx
          wrk = wrk + w(:, s(i, t-1))
       end do
       ! print*, 'h', wrk

       ! sample from Boltzmann
       wrk = exp(wrk)
       do i = 1, n
          i1 = m_cumsum(i)+1; i2 = m_cumsum(i+1)
          do j = i1+1, i2
             wrk(j) = wrk(j) + wrk(j-1)
          end do
          j = 0
          do while(wrk(i1+j) <= wrk(i2) * randr(i,t))
             j = j + 1
          end do
          x(i, t) = j
       end do
       ! print*, 'x', x(:, t)

       ! one hot encoding
       do i = 1, n_idx
          s(i, t) = x(idx(idx_ptr(i)+1), t)
          do j = idx_ptr(i)+2, idx_ptr(i+1)
             s(i, t) = s(i, t) * m(idx(j)) + x(idx(j), t)
          end do
          s(i, t) = s(i, t) + stratifier(i)
       end do
       ! print*, 's', s(:, t)

    end do

    deallocate(bc, idx_ptr, idx, stratifier, s)

  end subroutine simulate_time_series

  subroutine simulate_mutations(w, m_sum, n_s, n, m, l, degs, n_deg, x, y)
    ! simulate_mutations description

    real*8, intent(in) :: w(m_sum, n_s)
    integer, intent(in) :: m_sum, n_s, n, n_deg, l, m(n), degs(n_deg)
    integer, intent(out) :: x(n,l), y(n,l)
    !f2py integer, intent(hide), depend(w) :: m_sum=shape(w,0)
    !f2py integer, intent(hide), depend(w) :: n_s=shape(w,1)
    !f2py integer, intent(hide), depend(m) :: n=len(m)
    !f2py integer, intent(hide), depend(degs) :: n_deg=len(degs)

    integer i, i1, i2, j, k, max_deg, n_idx, n_idx_by_deg(n_deg), m_cumsum(n+1)
    real*8 randx(n, l), randy(n, l), wrk(m_sum)
    integer, allocatable :: bc(:), idx_ptr(:), idx(:), stratifier(:), s(:,:)

    call random_seed()
    call random_number(randx)
    call random_number(randy)

    max_deg = degs(n_deg)
    allocate(bc(max_deg))
    bc(1) = n
    do i = 2, max_deg
       bc(i) = bc(i-1)*(n-i+1)/i
    end do
    n_idx_by_deg = (/ (bc(degs(i)), i=1, n_deg) /)
    n_idx = sum(n_idx_by_deg)
    m_cumsum(1) = 0
    do i = 1, n
       m_cumsum(i+1) = m_cumsum(i) + m(i)
    end do

    allocate(idx_ptr(n_idx+1))
    allocate(idx(sum(n_idx_by_deg*degs)))
    allocate(stratifier(n_idx))

    call multiindices(n, m, degs, n_deg, n_idx, n_idx_by_deg, idx_ptr, idx, stratifier)

    ! random initial condition
    do i = 1, n
       do j = 1, l
          x(i, j) = mod(int((10*m(i)+1)*randx(i, j)), m(i))
       end do
    end do

    allocate(s(n_idx, l))
    do i = 1, n_idx
       do j = 1, l
          s(i, j) = x(idx(idx_ptr(i)+1), j)
          do k = idx_ptr(i)+2, idx_ptr(i+1)
             s(i, j) = s(i, j) * m(idx(k)) + x(idx(k), j)
          end do
          s(i, j) = s(i, j) + stratifier(i)
       end do
    end do
    ! print*, 'initial x', x(:, 1)

    do k = 1, l

       ! print*, 'k', k

       ! energies
       wrk = 0
       do i = 1, n_idx
          wrk = wrk + w(:, s(i, k))
       end do
       ! print*, 'h', wrk

       ! sample from Boltzmann
       wrk = exp(wrk)
       do i = 1, n
          i1 = m_cumsum(i)+1; i2 = m_cumsum(i+1)
          do j = i1+1, i2
             wrk(j) = wrk(j) + wrk(j-1)
          end do
          j = 0
          do while(wrk(i1+j) <= wrk(i2) * randy(i, k))
             j = j + 1
          end do
          y(i, k) = j
       end do
       ! print*, 'y', y(:, k)

    end do

    deallocate(bc, idx_ptr, idx, stratifier, s)

  end subroutine simulate_mutations

  subroutine discrete_fit(x, y, n_x, n_y, m_x, m_y, m_y_sum, l, degs, n_deg, &
       x_oh_pinv1, x_oh_pinv2, x_oh_pinv3, x_oh_rank, n_s, &
       iters, overfit, impute, w, disc, iter)
    implicit none

    integer, intent(in) :: n_x, n_y, l, n_deg, x_oh_rank, n_s, m_y_sum
    integer, intent(in) :: x(n_x,l), y(n_y,l), m_x(n_x), m_y(n_y), degs(n_deg)
    real*8, intent(in) :: x_oh_pinv1(l, x_oh_rank)
    real*8, intent(in) :: x_oh_pinv2(x_oh_rank)
    real*8, intent(in) :: x_oh_pinv3(x_oh_rank, n_s)
    integer, intent(in) :: iters
    logical, intent(in) :: overfit, impute
    real*8, intent(out) :: w(m_y_sum, n_s), disc(n_y, iters)
    integer, intent(out) :: iter(n_y)
    !f2py integer, intent(hide), depend(x) :: n_x=shape(x,0)
    !f2py integer, intent(hide), depend(y) :: n_y=shape(y,0)
    !f2py integer, intent(hide), depend(x) :: l=shape(x,1)
    !f2py integer, intent(hide), depend(degs) :: n_deg=len(degs)
    !f2py integer, intent(hide), depend(x_oh_pinv2) :: x_oh_rank=len(x_oh_pinv2)
    !f2py integer, intent(hide), depend(x_oh_pinv3) :: n_s=shape(x_oh_pinv3,1)

    integer i, i1, i2, j, k, max_deg, n_idx, n_idx_by_deg(n_deg), m_y_cumsum(n_y+1)
    integer, allocatable :: bc(:), idx_ptr(:), idx(:), var(:), stratifier(:), s(:,:)

    ! print*, 'x_oh_pinv1', shape(x_oh_pinv1)
    ! print*, 'x_oh_pinv2', shape(x_oh_pinv2)
    ! print*, 'x_oh_pinv3', shape(x_oh_pinv3)
    ! print*, 'n_x', n_x
    ! print*, 'l', l

    ! print*, 'x'
    ! do i = 1, n_x
    !    print*, x(i, :5)
    ! end do

    ! max_deg = degs(n_deg)
    ! allocate(bc(max_deg))
    ! bc(1) = n_x
    ! do i = 2, max_deg
    !    bc(i) = bc(i-1)*(n_x-i+1)/i
    ! end do
    ! n_idx_by_deg = (/ (bc(degs(i)), i=1, n_deg) /)
    ! n_idx = sum(n_idx_by_deg)
    ! allocate(idx_ptr(n_idx+1))
    ! allocate(idx(sum(n_idx_by_deg*degs)))
    ! allocate(stratifier(n_idx))

    ! call multiindices(n_x, m_x, degs, n_deg, n_idx, n_idx_by_deg, idx_ptr, idx, stratifier)

    max_deg = degs(n_deg)
    ! print*, 'n_deg', n_deg
    ! print*, 'max_deg', max_deg

    ! binomial coeff
    allocate(bc(max_deg))
    bc(1) = n_x
    ! bc(2:) = (/ (bc(i-1)*(n_x-i+1)/i, i=2, max_deg) /)
    do i = 2, max_deg
       bc(i) = bc(i-1)*(n_x-i+1)/i
    end do
    ! print*, 'bc', bc

    n_idx_by_deg = (/ (bc(degs(i)), i=1, n_deg) /)
    n_idx = sum(n_idx_by_deg)
    ! print*, 'n_idx_by_deg', n_idx_by_deg
    ! print*, 'n_idx', n_idx

    allocate(idx_ptr(n_idx+1))
    idx_ptr(1) = 0
    i = 2
    do j = 1, n_deg
       ! do k = 1, n_idx_by_deg(degs(j))
       do k = 1, n_idx_by_deg(j)
          idx_ptr(i) = idx_ptr(i-1) + degs(j)
          i = i + 1
       end do
    end do
    ! print*, 'idx_ptr', idx_ptr

    allocate(idx(sum(n_idx_by_deg*degs)))
    i = 1
    do j = 1, n_deg
       allocate(var(degs(j)))
       ! var = (/ (i, i=1, degs(j)) /)
       do i2 = 1, degs(j)
          var(i2) = i2
       end do
       idx(idx_ptr(i)+1:idx_ptr(i+1)) = var
       i = i + 1
       ! do k = 1, n_idx_by_deg(degs(j))-1
       do k = 1, n_idx_by_deg(j)-1
          i1 = degs(j)
          var(i1) = var(i1) + 1
          do while (var(i1) + degs(j) > i1 + n_x)
             i1 = i1 - 1
             var(i1) = var(i1) + 1
          end do
          var(i1+1:) = (/ (var(i2-1)+1, i2=i1+1, degs(j)) /)
          idx(idx_ptr(i)+1:idx_ptr(i+1)) = var
          i = i + 1
       end do
       deallocate(var)
    end do
    ! print*, 'idx', idx

    allocate(stratifier(n_idx))
    stratifier = 1
    do i = 2, n_idx
       do j = idx_ptr(i-1)+1, idx_ptr(i)
          stratifier(i) = stratifier(i) * m_x(idx(j))
       end do
       stratifier(i) = stratifier(i) + stratifier(i-1)
    end do
    ! print*, 'stratifier', stratifier

    ! powers of x stratified, i.e. cols of w
    allocate(s(n_idx, l))
    do i = 1, n_idx
       s(i, :) = x(idx(idx_ptr(i)+1), :)
       do j = idx_ptr(i)+2, idx_ptr(i+1)
          s(i, :) = s(i, :)*m_x(idx(j)) + x(idx(j), :)
       end do
       s(i, :) = s(i, :) + stratifier(i)
    end do
    ! print*, 's'
    ! do i = 1, n_idx
    !    print*, s(i, :5)
    ! end do

    m_y_cumsum(1) = 0
    do i = 1, n_y
       m_y_cumsum(i+1) = m_y_cumsum(i) + m_y(i)
    end do

    ! print*, 'm_y_cumsum', m_y_cumsum
    ! print*, 'n_s', n_s
    ! print*, 'w', shape(w)

    !$omp parallel do default(shared) private(i1,i2)
    do i = 1, n_y
       i1 = m_y_cumsum(i)+1
       i2 = m_y_cumsum(i+1)
       call discrete_fit_i(i1, i2, s, n_idx, y(i,:)+1, m_y(i), l, &
            x_oh_pinv1, x_oh_pinv2, x_oh_pinv3, x_oh_rank, n_s, &
            iters, overfit, impute, w(i1:i2,:), disc(i,:), iter(i))
    end do
    !$omp end parallel do

    deallocate(bc, idx_ptr, idx, stratifier, s)

  end subroutine discrete_fit

  subroutine discrete_fit_i(i1, i2, s, n_idx, y, m_y, l, &
       x_oh_pinv1, x_oh_pinv2, x_oh_pinv3, x_oh_rank, n_s, &
       iters, overfit, impute, w, disc, iter)
    implicit none

    integer, intent(in) :: i1, i2, n_idx, m_y, l, x_oh_rank, n_s, iters
    integer, intent(in) :: s(n_idx, l), y(l)
    real*8, intent(in) :: x_oh_pinv1(l, x_oh_rank)
    real*8, intent(in) :: x_oh_pinv2(x_oh_rank)
    real*8, intent(in) :: x_oh_pinv3(x_oh_rank, n_s)
    logical, intent(in) :: overfit, impute

    real*8, intent(out) :: w(m_y, n_s), disc(iters)
    integer, intent(out) :: iter

    logical disc_mask(m_y, l)
    integer n_wrong_states
    real*8 wrk(m_y, l), dw(m_y, x_oh_rank)
    integer i, t

    w = 0
    disc(1) = 1.0 / m_y / m_y + 1.0
    n_wrong_states = (m_y - 1) * l

    disc_mask = .true.
    do t = 1, l
       disc_mask(y(t), t) = .false.
    end do

    do iter = 2, iters

       ! compute energies
       wrk = 0
       do t = 1, l
          do i = 1, n_idx
             wrk(:, t) = wrk(:, t) + w(:, s(i, t))
          end do
       end do

       ! probabilities
       wrk = exp(wrk)
       do t = 1, l
          wrk(:, t) = wrk(:, t) / sum(wrk(:, t))
       end do

       ! discrepancy
       disc(iter) = sum(wrk*wrk, disc_mask) / n_wrong_states
       if (.not.overfit.and.(disc(iter) > disc(iter-1))) then
          exit
       end if

       do t = 1, l
          wrk(y(t), t) = wrk(y(t), t) - 1
       end do

       dw = matmul(wrk, x_oh_pinv1)
       do i = 1, x_oh_rank
          dw(:, i) = dw(:,i) * x_oh_pinv2(i)
       end do

       w = w - matmul(dw, x_oh_pinv3)

       if (impute) then
          w(:, i1:i2) = 0
       end if

    end do

    iter = iter - 1

  end subroutine discrete_fit_i

  subroutine continuous_fit(x, y, n, l, iters, atol, rtol, impute, w, disc, iter)
    implicit none

    real*8, intent(in) ::  x(n,l), y(n,l), atol, rtol
    integer, intent(in) :: n, l, iters
    real*8, intent(out) :: w(n,n), disc(n,iters)
    integer, intent(out) :: iter(n)
    logical, intent(in) :: impute
    real*8 symx(n,l), mean_x(n), x_mean0(n,l), cov_x(n,n)
    real*8 dt, sqrt_dt, sqrt_2, rat
    !f2py integer, intent(hide), depend(x) :: n=shape(x,0)
    !f2py integer, intent(hide), depend(l) :: n=shape(x,1)
    integer i, j, info, ipiv(n)

    dt = 1.0
    sqrt_dt = sqrt(dt)
    sqrt_2 = sqrt(2.0)
    rat = sqrt_dt / sqrt_2

    symx = 1
    symx = sign(symx, y-x)
    ! print*, 'symx(:,1)', symx(:,1)

    do i = 1, n
       mean_x(i) = sum(x(i,:))
    end do
    mean_x = mean_x / l
    ! print*, 'mean_x', mean_x

    do i = 1, l
       x_mean0(:,i) = x(:,i) - mean_x
    end do
    ! print*, 'x_mean0(:,1)', x_mean0(:,1)

    do i = 1, n
       do j = 1, n
          cov_x(i,j) = dot_product(x(i,:), x(j,:))
       end do
    end do
    cov_x = cov_x / l
    ! print*, 'cov_x', cov_x

    call dgetrf(n, n, cov_x, n, ipiv, info)
    ! print*, 'dgetrf info', info

    !$omp parallel do default(shared) private(i)
    do i = 1, n
       ! print*, 'fit_i', i
       call continuous_fit_i(i, x, n, l, x_mean0, symx(i,:), cov_x, ipiv, &
            iters, atol, rtol, impute, w(i,:), disc(i,:), iter(i))
    end do
    !$omp end parallel do

    w = w / rat

  end subroutine continuous_fit

  subroutine continuous_fit_i(i, x, n, l, x_mean0, symx, cov_x, ipiv, iters, atol, rtol, impute, w, disc, iter)
    implicit none

    real*8, intent(in) :: x(n,l), x_mean0(n,l), symx(l), cov_x(n,n), atol, rtol
    integer, intent(in) :: i, n, l, iters, ipiv(n)
    logical, intent(in) :: impute
    real*8, intent(out) :: w(n), disc(iters)
    integer, intent(out) :: iter
    real*8 h(l), erf_last(l), erf_next(l)
    integer info

    w = 0
    w(i) = 1
    erf_last = erf(x(i,:)) + 1

    do iter = 1, iters

       h = matmul(w,x)

       erf_next = erf(h)

       disc(iter) = norm2(erf_next-erf_last)
       if ((disc(iter) < atol).or.(disc(iter) < norm2(erf_next)*rtol)) exit
       erf_last = erf_next

       h = h * symx / erf_next

       w = matmul(x_mean0, h) / l
       call dgetrs('N', n, 1, cov_x, n, ipiv, w, n, info)

       if (impute) w(i) = 0

    end do

    iter = iter - 1

  end subroutine continuous_fit_i

end module fortran_module
