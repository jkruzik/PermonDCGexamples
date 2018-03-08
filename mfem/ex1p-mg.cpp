//                       MFEM Example 1 - Parallel Version
//                              PETSc Modification
//
// Compile with: make ex1p
//
// Sample runs:
//    mpirun -np 4 ex1p -m ../../data/amr-quad.mesh
//    mpirun -np 4 ex1p -m ../../data/amr-quad.mesh --petscopts rc_ex1p
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.
//               The example also shows how PETSc Krylov solvers can be used by
//               wrapping a HypreParMatrix (or not) and a Solver, together with
//               customization using an options file (see rc_ex1p) We also
//               provide an example on how to visualize the iterative solution
//               inside a PETSc solver.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <permonksp.h>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

// Geometric Multigrid
class Multigrid : public Solver
{
public:
    Multigrid(HypreParMatrix &Operator,
              const Array<HypreParMatrix*> &P,
              Solver *CoarsePrec=NULL)
        :
          Solver(Operator.GetNumRows()),
          P_(P),
          Operators_(P.Size()+1),
          Smoothers_(Operators_.Size()),
          current_level(Operators_.Size()-1),
          correction(Operators_.Size()),
          residual(Operators_.Size()),
          CoarseSolver(NULL),
          CoarsePrec_(CoarsePrec)
    {
        if (CoarsePrec)
        {
            CoarseSolver = new CGSolver(Operators_[0]->GetComm());
            CoarseSolver->SetRelTol(1e-8);
            CoarseSolver->SetMaxIter(50);
            CoarseSolver->SetPrintLevel(0);
            CoarseSolver->SetOperator(*Operators_[0]);
            CoarseSolver->SetPreconditioner(*CoarsePrec);
        }

        Operators_.Last() = &Operator;
        HypreParMatrix *AP;
        for (int l = Operators_.Size()-1; l > 0; l--)
        {
            // Two steps RAP
            //unique_ptr<HypreParMatrix> PT( P[l-1]->Transpose() );
            //unique_ptr<HypreParMatrix> AP( ParMult(Operators_[l], P[l-1]) );
            //Operators_[l-1] = ParMult(PT.get(), AP.get());
            AP = ParMult(Operators_[l], P[l-1]);
            Operators_[l-1] = ParMult(P[l-1]->Transpose(), AP);
            Operators_[l-1]->CopyRowStarts();
        }

        for (int l = 0; l < Operators_.Size(); l++)
        {
            Smoothers_[l] = new HypreSmoother(*Operators_[l]);
            correction[l] = new Vector(Operators_[l]->GetNumRows());
            residual[l] = new Vector(Operators_[l]->GetNumRows());
        }
    }

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op) { }

    ~Multigrid()
    {
        for (int l = 0; l < Operators_.Size(); l++)
        {
            delete Smoothers_[l];
            delete correction[l];
            delete residual[l];
        }
    }

private:
    void MG_Cycle() const;

    const Array<HypreParMatrix*> &P_;

    Array<HypreParMatrix*> Operators_;
    Array<HypreSmoother*> Smoothers_;

    mutable int current_level;

    mutable Array<Vector*> correction;
    mutable Array<Vector*> residual;

    mutable Vector res_aux;
    mutable Vector cor_cor;
    mutable Vector cor_aux;

    CGSolver *CoarseSolver;
    Solver *CoarsePrec_;
};

void Multigrid::Mult(const Vector & x, Vector & y) const
{
    *residual.Last() = x;
    correction.Last()->SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle();
}

void Multigrid::MG_Cycle() const
{
    // PreSmoothing
    const HypreParMatrix& Operator_l = *Operators_[current_level];
    const HypreSmoother& Smoother_l = *Smoothers_[current_level];

    Vector& residual_l = *residual[current_level];
    Vector& correction_l = *correction[current_level];

    Smoother_l.Mult(residual_l, correction_l);
    Operator_l.Mult(-1.0, correction_l, 1.0, residual_l);

    // Coarse grid correction
    if (current_level > 0)
    {
        const HypreParMatrix& P_l = *P_[current_level-1];

        P_l.MultTranspose(residual_l, *residual[current_level-1]);

        current_level--;
        MG_Cycle();
        current_level++;

        cor_cor.SetSize(residual_l.Size());
        P_l.Mult(*correction[current_level-1], cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
    }
    else
    {
        cor_cor.SetSize(residual_l.Size());
        if (CoarseSolver)
        {
            CoarseSolver->Mult(residual_l, cor_cor);
            correction_l += cor_cor;
            Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
        }
    }

    // PostSmoothing
    Smoother_l.Mult(residual_l, cor_cor);
    correction_l += cor_cor;
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  // 1. Initialize MPI.
  int num_procs, myid;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // 2. Parse command-line options.
  const char *mesh_file = "../../data/star.mesh";
  int order = 1;
  bool static_cond = false;
  bool visualization = false;
  bool use_petsc = true;
  const char *petscrc_file = "";
  bool petscmonitor = false;
  double nelems = 1000;
  int par_ref_levels = 2;
  bool dcgmg = false;
  bool mg = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                 "--no-static-condensation", "Enable static condensation.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                 "--no-petsc",
                 "Use or not PETSc to solve the linear system.");
  args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                 "PetscOptions file to use.");
  args.AddOption(&nelems, "-nelems", "--nelems",
                 "Max. number of serial mesh elements");
  args.AddOption(&par_ref_levels, "-par-ref-lvls", "--par-ref-lvls",
                 "Number of paralel mesh refinement levels");
  args.AddOption(&dcgmg, "-dcgmg", "--dcgmg",
                 "-no-dcgmg", "--no-dcgmg",
                 "Use mg prologation as a deflation space");
  args.AddOption(&mg, "-mg", "--mg",
                 "-no-mg", "--no-mg",
                 "Use PCMG");
  args.Parse();
  //if (!args.Good())
  //{
  //   if (myid == 0)
  //   {
  //      args.PrintUsage(cout);
  //   }
  //   MPI_Finalize();
  //   return 1;
  //}
  if (myid == 0)
  {
     args.PrintOptions(cout);
  }

  // 2b. We initialize PERMON
  PermonInitialize(&argc,&argv,petscrc_file,NULL);

  // 3. Read the (serial) mesh from the given mesh file on all processors.  We
  //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
  //    and volume meshes with the same code.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  // 4. Refine the serial mesh on all processors to increase the resolution. In
  //    this example we do 'ref_levels' of uniform refinement. We choose
  //    'ref_levels' to be the largest number that gives a final mesh with no
  //    more than 10,000 elements.
  {
     int ref_levels =
        (int)floor(log(nelems/mesh->GetNE())/log(2.)/dim);
     for (int l = 0; l < ref_levels; l++)
     {
        mesh->UniformRefinement();
     }
  }

  // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
  //    this mesh further in parallel to increase the resolution. Once the
  //    parallel mesh is defined, the serial mesh can be deleted.
  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;
  //{
  //   for (int l = 0; l < par_ref_levels; l++)
  //   {
  //      pmesh->UniformRefinement();
  //   }
  //}

  FiniteElementCollection *fec = new H1_FECollection(order, dim);
  ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
  
  // *** Compute prolongation
  PetscLogStage preproc,setup,solve;
  PetscLogEvent prolongAssemb,petscMatAssemb,operAssemb;
  ierr = PetscLogStageRegister("MFEM preproc",&preproc);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MFEM setup",&setup);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MFEM solve",&solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("prolongAssemb",0,&prolongAssemb);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("pmatAssemb",0,&petscMatAssemb);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("operAssemb",0,&operAssemb);CHKERRQ(ierr);

  HypreParMatrix* P_global;
  PetscParMatrix* P_global_petsc;
	Array<HypreParMatrix*> Phypre(par_ref_levels);
  Mat *mats,*Amats,P;
  ierr = PetscMalloc1(par_ref_levels,&mats);CHKERRQ(ierr);

  auto coarse_fespace = new ParFiniteElementSpace(pmesh, fec);
  const SparseMatrix *P_local;

  ierr = PetscLogStagePush(preproc);CHKERRQ(ierr);
  for (int l = 0; l < par_ref_levels; l++)
  {
    coarse_fespace->Update();
    pmesh->UniformRefinement();
    ierr = PetscLogEventBegin(prolongAssemb,0,0,0,0);CHKERRQ(ierr);
    P_local = ((const SparseMatrix*)fespace->GetUpdateOperator());

    auto d_td_coarse = coarse_fespace->Dof_TrueDof_Matrix();
    auto RP_local = Mult(*fespace->GetRestrictionMatrix(), *P_local);
    P_global = d_td_coarse->LeftDiagMult(*RP_local,
                                       fespace->GetTrueDofOffsets());
    P_global->CopyColStarts();
    P_global->CopyRowStarts();
    Phypre[l] = P_global;
    ierr = PetscLogEventEnd(prolongAssemb,0,0,0,0);CHKERRQ(ierr);

    ierr = PetscLogEventBegin(petscMatAssemb,0,0,0,0);CHKERRQ(ierr);
    P_global_petsc = new PetscParMatrix(P_global,Operator::PETSC_MATAIJ);
    ierr = PetscLogEventEnd(petscMatAssemb,0,0,0,0);CHKERRQ(ierr);
    mats[l] = (Mat)*P_global_petsc;
    delete RP_local;
  }
	if (dcgmg) {
    ierr = PetscLogEventBegin(operAssemb,0,0,0,0);CHKERRQ(ierr);
    Mat newmat;
    P = mats[par_ref_levels-1];
    for (int i=par_ref_levels-1; i>0; i--) {
      ierr = MatMatMult(P,mats[i-1],MAT_INITIAL_MATRIX,PETSC_DECIDE,&newmat);CHKERRQ(ierr);
      ierr = MatDestroy(&P);CHKERRQ(ierr);
      P = newmat ;
    }
    ierr = PetscLogEventEnd(operAssemb,0,0,0,0);CHKERRQ(ierr);
  }
  delete coarse_fespace;
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  // 6. Define a parallel finite element space on the parallel mesh. Here we
  //    use continuous Lagrange finite elements of the specified order. If
  //    order < 1, we instead use an isoparametric/isogeometric space.
  HYPRE_Int size = fespace->GlobalTrueVSize();
  if (myid == 0)
  {
     cout << "Number of finite element unknowns: " << size << endl;
  }

  // 7. Determine the list of true (i.e. parallel conforming) essential
  //    boundary dofs. In this example, the boundary conditions are defined
  //    by marking all the boundary attributes from the mesh as essential
  //    (Dirichlet) and converting them to a list of true dofs.
  Array<int> ess_tdof_list;
  if (pmesh->bdr_attributes.Size())
  {
     Array<int> ess_bdr(pmesh->bdr_attributes.Max());
     ess_bdr = 1;
     fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  // 8. Set up the parallel linear form b(.) which corresponds to the
  //    right-hand side of the FEM linear system, which in this case is
  //    (1,phi_i) where phi_i are the basis functions in fespace.
  ParLinearForm *b = new ParLinearForm(fespace);
  ConstantCoefficient one(1.0);
  b->AddDomainIntegrator(new DomainLFIntegrator(one));
  b->Assemble();

  // 9. Define the solution vector x as a parallel finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  ParGridFunction x(fespace);
  x = 0.0;

  // 10. Set up the parallel bilinear form a(.,.) on the finite element space
  //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //     domain integrator.
  ParBilinearForm *a = new ParBilinearForm(fespace);
  a->AddDomainIntegrator(new DiffusionIntegrator(one));

  // 11. Assemble the parallel bilinear form and the corresponding linear
  //     system, applying any necessary transformations such as: parallel
  //     assembly, eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, static condensation, etc.
  if (static_cond) { a->EnableStaticCondensation(); }
  a->Assemble();

  Vector B, X;
  HypreParMatrix A;
  if (mg) {
    a->FormLinearSystem(ess_tdof_list,x,*b,A,X,B);
    Multigrid *precond =  new Multigrid(A,Phypre);
    CGSolver *pcg = new CGSolver(MPI_COMM_WORLD);
    pcg->SetRelTol(1e-6);
    pcg->SetAbsTol(1e-50);
    pcg->SetMaxIter(2000);
    pcg->SetOperator(A);
    pcg->SetPreconditioner(*precond);
    ierr = PetscLogStagePush(solve);CHKERRQ(ierr);
    pcg->Mult(B, X);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    if (!myid) {
		  if (pcg->GetConverged()) {
      	cout << "CG converged in " << pcg->GetNumIterations()<<endl;
      } else {
          cout << "CG did not converge in " << pcg->GetNumIterations() <<endl;
		  }
    }
    delete pcg;
    delete precond;
    }
  {
    Vector B2(B);
    Vector X2(X);
    PetscParMatrix A2(&A,Operator::PETSC_MATAIJ);

    if (myid == 0)
    {
       cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
    }

    // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
    //     preconditioner from hypre.
    // If petscrc_file has been given, we convert the HypreParMatrix to a
    // PetscParMatrix; the user can then experiment with PETSc command line
    // options.
    PetscLinearSolver *pcg = new PetscLinearSolver(A2);
    pcg->SetTol(1e-6);
    pcg->SetAbsTol(1e-50);
    pcg->SetMaxIter(2000);

    PC pc;
    KSP ksp = (KSP)*pcg;
    KSPSetOptionsPrefix(ksp,"s_");
    if (dcgmg) {
      ierr = KSPSetType(ksp,KSPDCG);CHKERRQ(ierr);
      ierr =  KSPDCGSetDeflationSpace(ksp,P);CHKERRQ(ierr);
    //} else if (mg) {
    //  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    //  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
    //  ierr = PCMGSetLevels(pc,par_ref_levels+1,NULL);CHKERRQ(ierr);
    //  for (l = 1;l<par_ref_levels+1;l++) ierr = PCMGSetInterpolation(pc,l,mats[l-1]);CHKERRQ(ierr);
    //  TODO: get KSPs using PCMGGetSmoother() and set MATs
    }
    ierr = PetscLogStagePush(setup);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);

    ierr = PetscLogStagePush(solve);CHKERRQ(ierr);
    pcg->Mult(B2, X2);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    PetscParVector *pxcg = new PetscParVector(PETSC_COMM_WORLD,X);
    pxcg->PlaceArray(X);
    Vec x = (Vec)*pxcg;

    PetscParVector *pbcg = new PetscParVector(PETSC_COMM_WORLD,B);
    pbcg->PlaceArray(B);
    Vec pb = (Vec)*pbcg;

    Mat mat = Mat(A2);
    PetscParVector *pxcg2 = new PetscParVector(PETSC_COMM_WORLD,X2);
    pxcg2->PlaceArray(X2);
    Vec x2 = (Vec)*pxcg2;
    
    Vec v;
    VecDuplicate(pb,&v);
    PetscReal norm;
    MatMult(mat,x,v);
    VecAXPY(v,-1.,pb);
    VecNorm(v,NORM_2,&norm);
    printf("rmg %.16f\n",norm);
    MatMult(mat,x2,v);
    VecAXPY(v,-1.,pb);
    VecNorm(v,NORM_2,&norm);
    printf("cmg %.16f\n",norm);
    delete pcg;
  }

  // 13. Recover the parallel grid function corresponding to X. This is the
  //     local finite element solution on each processor.
  a->RecoverFEMSolution(X, *b, x);

  // 14. Save the refined mesh and the solution in parallel. This output can
  //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
  //{
  //   ostringstream mesh_name, sol_name;
  //   mesh_name << "mesh." << setfill('0') << setw(6) << myid;
  //   sol_name << "sol." << setfill('0') << setw(6) << myid;

  //   ofstream mesh_ofs(mesh_name.str().c_str());
  //   mesh_ofs.precision(8);
  //   pmesh->Print(mesh_ofs);

  //   ofstream sol_ofs(sol_name.str().c_str());
  //   sol_ofs.precision(8);
  //   x.Save(sol_ofs);
  //}

  // 15. Send the solution by socket to a GLVis server.
  if (visualization)
  {
     char vishost[] = "localhost";
     int  visport   = 19916;
     socketstream sol_sock(vishost, visport);
     sol_sock << "parallel " << num_procs << " " << myid << "\n";
     sol_sock.precision(8);
     sol_sock << "solution\n" << *pmesh << x << flush;
  }

  // 16. Free the used memory.
  delete a;
  delete b;
  delete fespace;
  if (order > 0) { delete fec; }
  delete pmesh;

  // We finalize PETSc
  PermonFinalize();
  MPI_Finalize();

  return 0;
}
