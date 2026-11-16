! Mark the code a harness measures, so MIMIR times and weighs that alone (Fortran).
!
!     use mimir_region
!     call make_inputs()          ! not measured
!     call mimir_region_begin()
!     call solve()                ! measured: time_s, region_peak_rss_mb, region_extra_mb
!     call mimir_region_end()
!     call check()                ! not measured
!
! Outside MIMIR (no MIMIR_REGION), both do nothing. Several regions in turn add their
! times. Plain Fortran 2008: compile it with the harness.
!
! A GPU's work is asynchronous: compiled as CUDA Fortran (nvfortran -cuda), each mark
! first waits for the device and the region is the profiler's range, which Nsight
! records alone; compiled with OpenACC, it waits for the queues. A Fortran harness that
! calls CUDA C through a binding waits itself (cudaDeviceSynchronize) before each mark.
module mimir_region
  implicit none
  private
  public :: mimir_region_begin, mimir_region_end
  ! newunit= gives negative unit numbers: whether the channel is open is kept apart.
  integer, save :: unit_out, unit_ack
  logical, save :: tried = .false., open_ = .false.
contains
  subroutine mark(word)
!@cuf use cudafor
    character(len=*), intent(in) :: word
!@cuf integer :: istat
    character(len=4096) :: path
    character(len=1) :: c
    integer :: n, stat
    logical :: there
    if (.not. tried) then
      tried = .true.
      call get_environment_variable("MIMIR_REGION", path, n, stat)
      if (stat == 0 .and. n > 0) then
        ! A rank on another node finds no FIFO there and gives up.
        inquire(file=path(1:n), exist=there)
        if (there) then
          open(newunit=unit_out, file=path(1:n), access="stream", form="unformatted", &
               action="write", status="old", iostat=stat)
          if (stat == 0) open(newunit=unit_ack, file=path(1:n)//".ack", access="stream", &
                              form="unformatted", action="read", status="old", iostat=stat)
          open_ = stat == 0
        end if
      end if
    end if
    if (open_) then
!$acc wait
!@cuf istat = cudaDeviceSynchronize()
!@cuf if (word == "end") istat = cudaProfilerStop()
      write(unit_out, iostat=stat) word//achar(10)
      flush(unit_out)
      read(unit_ack, iostat=stat) c
!@cuf if (word == "begin") istat = cudaProfilerStart()
    end if
  end subroutine mark

  subroutine mimir_region_begin()
    call mark("begin")
  end subroutine mimir_region_begin

  subroutine mimir_region_end()
    call mark("end")
  end subroutine mimir_region_end
end module mimir_region
