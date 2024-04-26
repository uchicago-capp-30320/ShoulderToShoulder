import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

// guards
import { canActivateProfile } from './_guards/auth.guard';
// import { AdminGuard } from './guards/admin.guard';

// Components
import { HomepageComponent } from './homepage/homepage.component';
import { SignupPageComponent } from './signup-page/signup-page.component';
import { OnboardingComponent } from './onboarding/onboarding.component';
import { LogInComponent } from './log-in/log-in.component';

// Routes
const appRoutes: Routes = [
    { path: 'home', component: HomepageComponent },
    { path: 'sign-up', component: SignupPageComponent },
    { path: 'onboarding', component: OnboardingComponent, canActivate: [canActivateProfile]},
    { path: 'log-in', component: LogInComponent },
    { path: '', redirectTo: '/home', pathMatch: 'full' }
]

@NgModule({
  imports: [
    RouterModule.forRoot(appRoutes,
      {
        enableTracing: false, // <-- debugging purposes only
        scrollPositionRestoration: 'enabled',
      })
  ],
  exports: [RouterModule]
})
export class AppRoutingModule { }