import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

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