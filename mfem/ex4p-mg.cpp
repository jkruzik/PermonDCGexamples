//                       MFEM Example 3 - Parallel Version
//                              PETSc Modification
//
// Compile with: make ex3p
//
// Sample runs:
//    mpirun -np 4 ex3p -m ../../data/klein-bottle.mesh -o 2 -f 0.1 --petscopts rc_ex3p
//    mpirun -np 4 ex3p -m ../../data/klein-bottle.mesh -o 2 -f 0.1 --petscopts rc_ex3p_bddc --nonoverlapping
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Static condensation is
//               also illustrated.
//
//               The example also show how to use the non-overlapping feature of
//               the ParBilinearForm class to obtain the linear operator in
//               a format suitable for the BDDC preconditioner in PETSc.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <permonksp.h>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

int main(int argc, char *argv[])
{
   PetscErrorCode ierr;
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/beam-tet.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = false;
   bool use_petsc = true;
   const char *petscrc_file = "";
   bool use_nonoverlapping = false;
   int ref_levels = 1;
   int par_ref_levels = 2;
   bool dcgmg = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
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
   args.AddOption(&use_nonoverlapping, "-nonoverlapping", "--nonoverlapping",
                  "-no-nonoverlapping", "--no-nonoverlapping",
                  "Use or not the block diagonal PETSc's matrix format "
                  "for non-overlapping domain decomposition.");
   args.AddOption(&ref_levels, "-ref-lvls", "--ref-lvls",
                  "Number of paralel mesh refinement levels");
   args.AddOption(&par_ref_levels, "-par-ref-lvls", "--par-ref-lvls",
                  "Number of paralel mesh refinement levels");
   args.AddOption(&dcgmg, "-dcgmg", "--dcgmg",
                  "-no-dcgmg", "--no-dcgmg",
                  "Number of paralel mesh refinement levels");
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
   // 2b. We initialize PETSc
   if (use_petsc) { PermonInitialize(&argc,&argv,petscrc_file,NULL); }
   kappa = freq * M_PI;

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      if (myid == 0)
      {
        cout << "Number of elems in input mesh: " << mesh->GetNE()
             << " dim: " << dim << endl;
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
      if (myid == 0)
      {
        cout << "Number of elems in serial mesh: " << mesh->GetNE() << endl;
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
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
   Mat *mats,P;
   ierr = PetscMalloc1(par_ref_levels,&mats);CHKERRQ(ierr);

   auto coarse_fespace = new ParFiniteElementSpace(pmesh, fec);
   const SparseMatrix *P_local;

   ierr = PetscLogStagePush(preproc);CHKERRQ(ierr);
   for (int l = 0; l < par_ref_levels+1; l++)
   {
       if (l > 0)
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
           ierr = PetscLogEventEnd(prolongAssemb,0,0,0,0);CHKERRQ(ierr);

           ierr = PetscLogEventBegin(petscMatAssemb,0,0,0,0);CHKERRQ(ierr);
           P_global_petsc = new PetscParMatrix(P_global,Operator::PETSC_MATAIJ);
           ierr = PetscLogEventEnd(petscMatAssemb,0,0,0,0);CHKERRQ(ierr);
           mats[l-1] = (Mat)*P_global_petsc;
           delete RP_local;
       }
   }
   ierr = PetscLogEventBegin(operAssemb,0,0,0,0);CHKERRQ(ierr);
   Mat newmat;
   P = mats[par_ref_levels-1];
   for (int i=par_ref_levels-1; i>0; i--) {
     ierr = MatMatMult(P,mats[i-1],MAT_INITIAL_MATRIX,PETSC_DECIDE,&newmat);CHKERRQ(ierr);
     ierr = MatDestroy(&P);CHKERRQ(ierr);
     P = newmat ;
   }
   ierr = PetscLogEventEnd(operAssemb,0,0,0,0);CHKERRQ(ierr);
   delete coarse_fespace;
   ierr = PetscLogStagePop();CHKERRQ(ierr);

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
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
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(sdim, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   VectorFunctionCoefficient E(sdim, E_exact);
   x.ProjectCoefficient(E);

   // 10. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   Vector B, X;
   if (!use_petsc)
   {
      HypreParMatrix A;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

      if (myid == 0)
      {
         cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
      }

      // 12. Define and apply a parallel PCG solver for AX=B with the AMS
      //     preconditioner from hypre.
      ParFiniteElementSpace *prec_fespace =
         (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
      HypreSolver *ams = new HypreAMS(A, prec_fespace);
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-10);
      pcg->SetMaxIter(500);
      pcg->SetPrintLevel(2);
      pcg->SetPreconditioner(*ams);
      pcg->Mult(B, X);
      delete pcg;
      delete ams;
   }
   else
   {
      PetscParMatrix A;
      a->SetOperatorType(use_nonoverlapping ?
                         Operator::PETSC_MATIS : Operator::PETSC_MATAIJ);
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

      if (myid == 0)
      {
         cout << "Size of linear system: " << A.M() << endl;
      }

      // 12. Define and apply a parallel PCG solver.
      ParFiniteElementSpace *prec_fespace =
         (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);

      PetscLinearSolver *pcg = new PetscLinearSolver(A);
      pcg->SetTol(1e-6);
      pcg->SetAbsTol(1e-50);
      pcg->SetMaxIter(2000);
      KSP ksp = (KSP)*pcg;
      KSPSetOptionsPrefix(ksp,"s_");
      if (dcgmg) {
        ierr = KSPSetType(ksp,KSPDCG);CHKERRQ(ierr);
        ierr =  KSPDCGSetDeflationSpace(ksp,P);CHKERRQ(ierr);
      }
      ierr = PetscLogStagePush(setup);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = KSPSetUp(ksp);CHKERRQ(ierr);
      ierr = PetscLogStagePop();CHKERRQ(ierr);

      ierr = PetscLogStagePush(solve);CHKERRQ(ierr);
      pcg->Mult(B, X);
      ierr = PetscLogStagePop();CHKERRQ(ierr);
      delete pcg;
   }

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 14. Compute and print the L^2 norm of the error.
   {
      double err = x.ComputeL2Error(E);
      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
      }
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
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

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 17. Free the used memory.
   delete a;
   delete sigma;
   delete muinv;
   delete b;
   delete fespace;
   delete fec;
   delete pmesh;

   // We finalize PETSc
   if (use_petsc) { PetscFinalize(); }

   MPI_Finalize();

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
