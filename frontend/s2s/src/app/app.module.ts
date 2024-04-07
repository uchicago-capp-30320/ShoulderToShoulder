import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { NgxMaskDirective, IConfig, provideEnvironmentNgxMask } from 'ngx-mask';
import {MatTooltipModule} from '@angular/material/tooltip';
import { MatButtonModule } from '@angular/material/button'; 
import { HashLocationStrategy, LocationStrategy } from '@angular/common';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';


// primeng
import { MenubarModule } from 'primeng/menubar';
import { BadgeModule } from 'primeng/badge';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';
import { PasswordModule } from 'primeng/password';
import { CardModule } from 'primeng/card';
import { DividerModule } from 'primeng/divider';
import { DropdownModule } from 'primeng/dropdown';
import { MultiSelectModule } from 'primeng/multiselect';
import { TooltipModule } from 'primeng/tooltip';
const maskConfig: Partial<IConfig> = {
  validation: false,
};

// Routing
import { AppRoutingModule } from './app-routing.module';

// Components
import { AppComponent } from './app.component';
import { HomepageComponent } from './homepage/homepage.component';
import { NavbarComponent } from './navbar/navbar.component';
import { AppHeaderComponent } from './app-header/app-header.component';
import { FooterComponent } from './footer/footer.component';
import { SignupPageComponent } from './signup-page/signup-page.component';
import { OnboardingComponent } from './onboarding/onboarding.component';
import { DemographicsSurveyComponent } from './demographics-survey/demographics-survey.component';
import { PreferencesSurveyComponent } from './preferences-survey/preferences-survey.component';
import { ScenariosSurveyComponent } from './scenarios-survey/scenarios-survey.component';



@NgModule({
  declarations: [
    AppComponent,
    HomepageComponent,
    NavbarComponent,
    AppHeaderComponent,
    FooterComponent,
    SignupPageComponent,
    OnboardingComponent,
    DemographicsSurveyComponent,
    PreferencesSurveyComponent,
    ScenariosSurveyComponent,
    // LoginComponent,
    // LandingComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    MenubarModule,
    BadgeModule,
    ButtonModule,
    InputTextModule,
    PasswordModule,
    CardModule,
    ReactiveFormsModule,
    NgxMaskDirective,
    MatTooltipModule,
    MatButtonModule,
    TooltipModule,
    DividerModule,
    DropdownModule,
    MultiSelectModule
  ],
  providers: [
    provideEnvironmentNgxMask(maskConfig),
    provideAnimationsAsync(),
    { provide: LocationStrategy, useClass: HashLocationStrategy }
    ],
  bootstrap: [AppComponent]
})
export class AppModule { }
