#include <permonksp.h>

int main(int argc,char **args)
{

    PetscErrorCode ierr;
    KSP ksp = NULL;
    PC  pc;
    Mat K   = NULL;
    Vec f   = NULL;
    Vec u   = NULL;
    MPI_Comm comm;
    PetscViewer fd;
    char file[PETSC_MAX_PATH_LEN];
    static PetscLogStage loadStage, prepStage, solvStage;
    PetscBool flg;

    ierr = PermonInitialize(&argc,&args,(char *)0,NULL);if (ierr) return ierr;
    comm = PETSC_COMM_WORLD;
    ierr = PetscOptionsGetString(NULL,NULL,"-f0",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f0 option");

    
    TRY( PetscLogStageRegister("load data", &loadStage) );
    TRY( PetscLogStagePush(loadStage) );
    {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
      ierr = MatCreate(comm,&K);CHKERRQ(ierr);
      ierr = MatLoad(K,fd);CHKERRQ(ierr);
      ierr = MatCreateVecs(K,&f,&u);CHKERRQ(ierr);
      ierr = PetscPushErrorHandler(PetscIgnoreErrorHandler,NULL);CHKERRQ(ierr);
      ierr = VecLoad(f,fd);
      ierr = PetscPopErrorHandler();CHKERRQ(ierr);
      if (ierr) {
        PetscInt m;
        TRY( VecGetSize(f,&m) );
        ierr = VecSet(f,sqrt(m));CHKERRQ(ierr);
      }
      ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
    }
    TRY( PetscLogStagePop() );
    

    TRY( PetscLogStageRegister("preprocess", &prepStage) );
    TRY( PetscLogStagePush(prepStage) );
    {
      TRY( KSPCreate(comm, &ksp) );
      TRY( KSPSetType(ksp, KSPCG) );
      TRY( KSPSetOperators(ksp, K, K) );
      TRY( KSPGetPC(ksp,&pc) );
      TRY( PCSetType(pc,PCNONE) );
      TRY( KSPSetOptionsPrefix(ksp,"s_") );
      TRY( KSPSetFromOptions(ksp) );
      TRY( KSPSetUp(ksp) );
    }
    TRY( PetscLogStagePop() );
    
    TRY( PetscLogStageRegister("solve", &solvStage) );
    TRY( PetscLogStagePush(solvStage) );
    {
        TRY( KSPSolve(ksp,f,u) );   
    }
    TRY( PetscLogStagePop() );
 
    TRY( KSPDestroy(&ksp) );
    TRY( VecDestroy(&u) );
    TRY( VecDestroy(&f) );
    TRY( MatDestroy(&K) );
    ierr = PermonFinalize();
    return ierr;

}

