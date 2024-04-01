import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

// Components
import { LoginComponent } from './login/login.component';
// import { LandingComponent } from './landing/landing.component';

// Routes
const appRoutes: Routes = [
    { path: 'login', component: LoginComponent },
    // { path: 'landing', component: LandingComponent },
    // { path: '', redirectTo: '/landing', pathMatch: 'full' }
]

@NgModule({
  imports: [
    RouterModule.forRoot(appRoutes)
  ],
  exports: [RouterModule]
})
export class AppRoutingModule { }