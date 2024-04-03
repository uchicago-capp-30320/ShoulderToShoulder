import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

// guards
// import { AuthGuard } from './guards/auth.guard';
// import { AdminGuard } from './guards/admin.guard';

// Components
import { HomepageComponent } from './homepage/homepage.component';
// import { LoginComponent } from './login/login.component';

// Routes
const appRoutes: Routes = [
    // { path: 'login', component: LoginComponent },
    { path: 'home', component: HomepageComponent },
    { path: '', redirectTo: '/home', pathMatch: 'full' }
]

@NgModule({
  imports: [
    RouterModule.forRoot(appRoutes)
  ],
  exports: [RouterModule]
})
export class AppRoutingModule { }